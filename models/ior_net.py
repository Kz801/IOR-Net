import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    MultiheadAttention,
    TransformerDecoder,
    TransformerDecoderLayer,
)

from pos_embed import (
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)

try:
    from flash_attention_class import FlashAttention
except ImportError:
    FlashAttention = None

try:
    from flash_attn.modules.mlp import FusedMLP
except ImportError:
    FusedMLP = None

try:
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
except ImportError:
    DropoutAddRMSNorm = None


class PrototypeGuidedRefinerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        saliency_temperature: float = 0.1,
    ):
        super().__init__()
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.saliency_temperature = saliency_temperature

        self.saliency_projector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.saliency_gate = nn.Parameter(torch.zeros(1))

    def forward(self, tgt, memory, prototypes=None, **kwargs):
        if prototypes is not None:
            tgt_aligned = F.normalize(self.saliency_projector(tgt), p=2, dim=-1)
            protos_norm = F.normalize(prototypes, p=2, dim=-1)
            similarity_map = torch.matmul(tgt_aligned, protos_norm.t())
            saliency_scores, _ = torch.max(similarity_map, dim=-1, keepdim=True)
            saliency_weights = F.softmax(
                saliency_scores / self.saliency_temperature, dim=1
            )
            gate_val = torch.sigmoid(self.saliency_gate)
            tgt_refined = tgt + (gate_val * (tgt * saliency_weights))
        else:
            tgt_refined = tgt

        attn_out = self.multihead_attn(
            query=tgt_refined, key=memory, value=memory, need_weights=False
        )[0]
        tgt = self.norm1(tgt_refined + self.dropout1(attn_out))

        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(ffn_out))
        return tgt


class PostNormRefinerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, tgt, memory, base_point=None, prototypes=None, **kwargs):
        attn_out = self.multihead_attn(
            query=tgt, key=memory, value=memory, need_weights=False
        )[0]
        tgt = self.norm1(tgt + self.dropout1(attn_out))

        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(ffn_out))
        return tgt


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
        force_fp32: bool = False,
    ):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = (
                x.float().mul_(self.gamma.float())
                if self.inplace
                else x.float() * self.gamma.float()
            )
            return out.to(dtype=output_type)
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attn: bool = False,
        causal: bool = False,
        norm_layer=nn.LayerNorm,
        qk_normalization: bool = False,
        use_fused_rmsnorm: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = bool(use_flash_attn and FlashAttention is not None)
        if self.use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = (
                self.q_norm(q.transpose(1, 2).flatten(-2, -1))
                .view(B_, N_, H_, D_)
                .transpose(1, 2)
            )
            k = (
                self.k_norm(k.transpose(1, 2).flatten(-2, -1))
                .view(B_, N_, H_, D_)
                .transpose(1, 2)
            )
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    def _flash_attn(self, x, key_padding_mask=None, need_weights: bool = False):
        qkv = rearrange(
            self.qkv(x), "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )
        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)
        context, _ = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        return self.proj_drop(outs)

    def forward(self, x):
        return self._flash_attn(x) if self.use_flash_attn else self._naive_attn(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_flash_attn: bool = False,
        use_fused_mlp: bool = False,
        fused_mlp_heuristic: int = 1,
        with_cp: bool = False,
        qk_normalization: bool = False,
        layerscale_no_force_fp32: bool = False,
        use_fused_rmsnorm: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash_attn=use_flash_attn,
            causal=False,
            norm_layer=norm_layer,
            qk_normalization=qk_normalization,
            use_fused_rmsnorm=use_fused_rmsnorm,
        )
        self.ls1 = (
            LayerScale(
                dim, init_values=init_values, force_fp32=(not layerscale_no_force_fp32)
            )
            if init_values
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp and FusedMLP is not None:
            self.mlp = FusedMLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                heuristic=fused_mlp_heuristic,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        self.ls2 = (
            LayerScale(
                dim, init_values=init_values, force_fp32=(not layerscale_no_force_fp32)
            )
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def forward(self, x, residual=None):
        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            assert residual is None
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x

        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual)
        return _inner_forward(x, residual=residual)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        num_frames: int = 8,
        tubelet_size: int = 1,
        norm_layer=None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.t_grid_size = self.grid_size[0]
        self.s_grid_size = self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(3).permute(0, 2, 3, 1)
        return self.norm(x)


class IORNet(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 14,
        img_size: int = 224,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.25,
        embed_dim: int = 1408,
        head_drop_path_rate: float = 0.0,
        num_heads: int = 16,
        mlp_ratio: float = 4.3637,
        init_values: float = 1e-5,
        qk_normalization: bool = True,
        depth: int = 40,
        use_flash_attn: bool = True,
        use_fused_rmsnorm: bool = True,
        use_fused_mlp: bool = True,
        fused_mlp_heuristic: int = 1,
        attn_pool_num_heads: int = 16,
        clip_embed_dim: int = 768,
        layerscale_no_force_fp32: bool = False,
        num_frames: int = 8,
        tubelet_size: int = 1,
        sep_pos_embed: bool = False,
        use_checkpoint: bool = False,
        checkpoint_num: int = 0,
        fc_drop_rate: float = 0.0,
        num_classes: int = 1000,
        max_iterations: int = 3,
        momentum_factor: float = 0.9,
        saliency_temperature: float = 0.1,
        isr_global_decoder_layers: int = 1,
        isr_local_decoder_layers: int = 1,
        isr_query_gen_layers: int = 1,
        isr_mem_refiner_layers: int = 3,
        isr_query_synth_layers: int = 1,
        isr_global_fuser_layers: int = 1,
    ):
        super().__init__()

        self.max_iterations = max_iterations
        self.momentum_factor = momentum_factor
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_fused_rmsnorm = bool(
            use_fused_rmsnorm and DropoutAddRMSNorm is not None
        )
        self.use_flash_attn = bool(use_flash_attn and FlashAttention is not None)

        if self.use_fused_rmsnorm:
            norm_layer = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
            norm_layer_simple = partial(RMSNorm, eps=1e-6)
        else:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            norm_layer_simple = partial(nn.LayerNorm, eps=1e-6)

        self.norm_layer_for_blocks = norm_layer
        norm_layer_decoder = partial(nn.LayerNorm, eps=1e-6)
        norm_layer_eps = 1e-6

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            norm_layer=norm_layer_simple if not self.use_fused_rmsnorm else None,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.patch_embed.s_grid_size, embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_embed.t_grid_size, embed_dim)
            )
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        with_cp_list = [use_checkpoint and idx < checkpoint_num for idx in range(depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=self.norm_layer_for_blocks,
                    drop_path=dpr[i],
                    init_values=init_values,
                    attn_drop=0.0,
                    use_flash_attn=self.use_flash_attn,
                    use_fused_mlp=use_fused_mlp,
                    fused_mlp_heuristic=fused_mlp_heuristic,
                    with_cp=with_cp_list[i],
                    qk_normalization=qk_normalization,
                    layerscale_no_force_fp32=layerscale_no_force_fp32,
                    use_fused_rmsnorm=self.use_fused_rmsnorm,
                )
                for i in range(depth)
            ]
        )

        etf_matrix = torch.randn(num_classes, embed_dim)
        nn.init.orthogonal_(etf_matrix)
        etf_matrix = F.normalize(etf_matrix, p=2, dim=-1)
        self.prototypes = nn.Parameter(etf_matrix)

        self.initial_queries = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        trunc_normal_(self.initial_queries, std=0.02)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(30.0))

        ffn_dim = embed_dim * 4

        self.norm_q_gen = norm_layer_decoder(embed_dim)
        self.norm_kv_gen = norm_layer_decoder(embed_dim)
        self.query_generator_norm = norm_layer_decoder(embed_dim)
        self.norm_cls = norm_layer_decoder(embed_dim)
        self.norm_mem = norm_layer_decoder(embed_dim)

        self.norm_q_L1_global = norm_layer_decoder(embed_dim)
        self.norm_kv_L1_global = norm_layer_decoder(embed_dim)
        self.norm_q_L2_local = norm_layer_decoder(embed_dim)
        self.norm_kv_L2_local = norm_layer_decoder(embed_dim)

        self.norm_q_mem_ref = norm_layer_decoder(embed_dim)
        self.norm_kv_mem_ref = norm_layer_decoder(embed_dim)
        self.memory_refiner_local_norm = norm_layer_decoder(embed_dim)

        self.norm_for_global_calc = norm_layer_decoder(embed_dim)
        self.norm_q_fuser = norm_layer_decoder(embed_dim)
        self.norm_kv_fuser = norm_layer_decoder(embed_dim)
        self.global_memory_fuser_norm = norm_layer_decoder(embed_dim)

        self.norm_q_synth_g = norm_layer_decoder(embed_dim)
        self.norm_kv_synth_g = norm_layer_decoder(embed_dim)
        self.norm_q_synth_l = norm_layer_decoder(embed_dim)
        self.norm_kv_synth_l = norm_layer_decoder(embed_dim)

        self.query_generator_layers = nn.ModuleList(
            [
                PostNormRefinerLayer(
                    embed_dim, attn_pool_num_heads, ffn_dim, 0.1, "gelu"
                )
                for _ in range(isr_query_gen_layers)
            ]
        )

        self.memory_refiner_local_layers = nn.ModuleList(
            [
                PrototypeGuidedRefinerLayer(
                    embed_dim,
                    attn_pool_num_heads,
                    ffn_dim,
                    0.1,
                    "gelu",
                    saliency_temperature=saliency_temperature,
                )
                for _ in range(isr_mem_refiner_layers)
            ]
        )

        self.global_memory_fuser_layers = nn.ModuleList(
            [
                PostNormRefinerLayer(
                    embed_dim, attn_pool_num_heads, ffn_dim, 0.1, "gelu"
                )
                for _ in range(isr_global_fuser_layers)
            ]
        )

        decoder_layer_template = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=attn_pool_num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
            layer_norm_eps=norm_layer_eps,
        )
        self.global_decoder = TransformerDecoder(
            decoder_layer_template, num_layers=isr_global_decoder_layers
        )
        self.local_decoder = TransformerDecoder(
            decoder_layer_template, num_layers=isr_local_decoder_layers
        )
        self.global_query_synthesizer = TransformerDecoder(
            decoder_layer_template, num_layers=isr_query_synth_layers
        )
        self.local_query_synthesizer = TransformerDecoder(
            decoder_layer_template, num_layers=isr_query_synth_layers
        )

        gating_input_dim = embed_dim * 4
        gating_output_dim = num_classes * 2
        self.gating_network = nn.Sequential(
            norm_layer_decoder(gating_input_dim),
            nn.Linear(gating_input_dim, gating_output_dim),
        )

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def init_pos_embed(self):
        if self.sep_pos_embed:
            pos_embed_spatial = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1],
                int(self.patch_embed.s_grid_size ** 0.5),
                cls_token=False,
            )
            self.pos_embed_spatial.data.copy_(
                torch.from_numpy(pos_embed_spatial).float().unsqueeze(0)
            )
            pos_embed_temporal = get_1d_sincos_pos_embed(
                self.pos_embed_temporal.shape[-1],
                self.patch_embed.t_grid_size,
                cls_token=False,
            )
            self.pos_embed_temporal.data.copy_(
                torch.from_numpy(pos_embed_temporal).float().unsqueeze(0)
            )
        else:
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.patch_embed.grid_size[1],
                self.patch_embed.grid_size[0],
                cls_token=True,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        norm_types = (
            (nn.LayerNorm, RMSNorm, DropoutAddRMSNorm)
            if DropoutAddRMSNorm is not None
            else (nn.LayerNorm, RMSNorm)
        )
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, norm_types):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(
            m, (TransformerDecoderLayer, PostNormRefinerLayer, PrototypeGuidedRefinerLayer)
        ):
            for sub_m in m.modules():
                if isinstance(sub_m, nn.Linear):
                    trunc_normal_(sub_m.weight, std=0.02)
                    if sub_m.bias is not None:
                        nn.init.constant_(sub_m.bias, 0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if hasattr(layer, "attn"):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer, "mlp"):
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        start_layer_id = len(self.blocks) + 1
        refiner_layers = []
        for decoder_stack in (
            self.global_decoder,
            self.local_decoder,
            self.global_query_synthesizer,
            self.local_query_synthesizer,
        ):
            if hasattr(decoder_stack, "layers"):
                refiner_layers.extend(decoder_stack.layers)
        for module_list in (
            self.query_generator_layers,
            self.memory_refiner_local_layers,
            self.global_memory_fuser_layers,
        ):
            refiner_layers.extend(module_list)

        for mod in refiner_layers:
            if hasattr(mod, "linear2"):
                rescale(mod.linear2.weight.data, start_layer_id)
            if hasattr(mod, "multihead_attn") and hasattr(
                mod.multihead_attn, "out_proj"
            ):
                rescale(mod.multihead_attn.out_proj.weight.data, start_layer_id)

    def _encode_backbone(self, x):
        x = self.patch_embed(x.type(self.dtype))
        B, T, L, C = x.shape
        x = x.view(B, T * L, C)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed_spatial = self.pos_embed_spatial.repeat(
                1, self.patch_embed.t_grid_size, 1
            )
            pos_embed_temporal = self.pos_embed_temporal.repeat_interleave(
                self.patch_embed.s_grid_size, dim=1
            )
            x_patches = x[:, 1:, :] + (pos_embed_spatial + pos_embed_temporal)
            x_cls = x[:, :1, :] + self.pos_embed_cls
            x = torch.cat((x_cls, x_patches), dim=1)
        else:
            x = x + self.pos_embed

        residual = None
        for blk in self.blocks:
            if self.use_fused_rmsnorm:
                if isinstance(x, tuple) and len(x) == 2:
                    x, residual = x
                x = blk(x, residual=residual)
            else:
                x = blk(x)

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        return self.norm_cls(x[:, 0, :]), self.norm_mem(x[:, 1:, :])

    @staticmethod
    def _orthogonal_decompose(state, prototypes_norm):
        proj_mag = torch.sum(
            state * prototypes_norm.unsqueeze(0), dim=-1, keepdim=True
        )
        projected_signal = proj_mag * prototypes_norm.unsqueeze(0)
        orthogonal_residual = state - projected_signal
        return projected_signal, orthogonal_residual

    def forward(self, x):
        x_cls_normed, patches_normed = self._encode_backbone(x)
        B = x_cls_normed.shape[0]

        tgt_init = self.norm_q_gen(self.initial_queries.expand(B, -1, -1))
        mem_init = self.norm_kv_gen(patches_normed)
        h_init = tgt_init
        for layer in self.query_generator_layers:
            h_init = layer(tgt=h_init, memory=mem_init)
        h_init = self.query_generator_norm(h_init)

        h_glo = h_init
        h_loc = h_init
        m_glo = x_cls_normed.unsqueeze(1)
        m_loc = patches_normed

        v_m_loc = torch.zeros_like(m_loc)
        v_m_glo = torch.zeros_like(m_glo)
        v_h_glo = torch.zeros_like(h_glo)
        v_h_loc = torch.zeros_like(h_loc)

        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        last_logits = None

        for k in range(self.max_iterations):
            p_glo, r_glo = self._orthogonal_decompose(h_glo, prototypes_norm)
            r_glo_refined = self.global_decoder(
                tgt=self.norm_q_L1_global(r_glo),
                memory=self.norm_kv_L1_global(m_glo),
            )
            q_glo = p_glo + r_glo_refined

            p_loc, r_loc = self._orthogonal_decompose(h_loc, prototypes_norm)
            r_loc_refined = self.local_decoder(
                tgt=self.norm_q_L2_local(r_loc),
                memory=self.norm_kv_L2_local(m_loc),
            )
            q_loc = p_loc + r_loc_refined

            q_glo_norm = F.normalize(q_glo, p=2, dim=-1)
            q_loc_norm = F.normalize(q_loc, p=2, dim=-1)
            scale_factor = self.logit_scale.exp()

            logits_glo = (
                torch.sum(q_glo_norm * prototypes_norm.unsqueeze(0), dim=-1)
                * scale_factor
            )
            logits_loc = (
                torch.sum(q_loc_norm * prototypes_norm.unsqueeze(0), dim=-1)
                * scale_factor
            )

            gate_input = torch.cat(
                [
                    x_cls_normed,
                    q_glo.mean(dim=1),
                    q_loc.mean(dim=1),
                    torch.abs(q_glo - q_loc).mean(dim=1),
                ],
                dim=-1,
            )
            gate_glo, gate_loc = torch.chunk(
                self.gating_network(gate_input), 2, dim=-1
            )
            last_logits = (
                torch.sigmoid(gate_glo) * logits_glo
                + torch.sigmoid(gate_loc) * logits_loc
            )

            if k < self.max_iterations - 1:
                delta_m_loc = self.norm_q_mem_ref(m_loc)
                feedback_loc = self.norm_kv_mem_ref(q_loc)
                for layer in self.memory_refiner_local_layers:
                    delta_m_loc = layer(
                        tgt=delta_m_loc,
                        memory=feedback_loc,
                        prototypes=self.prototypes,
                    )
                v_m_loc = (
                    self.momentum_factor * v_m_loc
                    + (1.0 - self.momentum_factor)
                    * self.memory_refiner_local_norm(delta_m_loc)
                )
                m_loc = m_loc + v_m_loc

                pooled_loc = self.norm_for_global_calc(m_loc).mean(dim=1, keepdim=True)
                delta_m_glo = self.norm_q_fuser(m_glo)
                mem_fuser = self.norm_kv_fuser(pooled_loc)
                for layer in self.global_memory_fuser_layers:
                    delta_m_glo = layer(tgt=delta_m_glo, memory=mem_fuser)
                v_m_glo = (
                    self.momentum_factor * v_m_glo
                    + (1.0 - self.momentum_factor)
                    * self.global_memory_fuser_norm(delta_m_glo)
                )
                m_glo = m_glo + v_m_glo

                delta_h_glo = self.global_query_synthesizer(
                    tgt=self.norm_q_synth_g(q_glo),
                    memory=self.norm_kv_synth_g(q_loc),
                )
                delta_h_loc = self.local_query_synthesizer(
                    tgt=self.norm_q_synth_l(q_loc),
                    memory=self.norm_kv_synth_l(q_glo),
                )
                v_h_glo = (
                    self.momentum_factor * v_h_glo
                    + (1.0 - self.momentum_factor) * delta_h_glo
                )
                v_h_loc = (
                    self.momentum_factor * v_h_loc
                    + (1.0 - self.momentum_factor) * delta_h_loc
                )
                h_glo = h_glo + v_h_glo
                h_loc = h_loc + v_h_loc

        return last_logits

@register_model
def ior_net_base_patch14_224(pretrained: bool = False, **kwargs):
    max_iterations = kwargs.pop("max_iterations", 3)
    return IORNet(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        max_iterations=max_iterations,
        **kwargs,
    )


@register_model
def ior_net_small_patch14_224(pretrained: bool = False, **kwargs):
    max_iterations = kwargs.pop("max_iterations", 3)
    return IORNet(
        img_size=224,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        max_iterations=max_iterations,
        **kwargs,
    )