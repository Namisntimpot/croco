import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.mlp import Mlp
from timm.layers.drop import DropPath
from typing import *

from .attention import AttentionLayer
from .utils import parse_norm_layer_1d, to_2tuple

class SelfAttnBlock(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: str = 'none',
            rope_func = None,
            get_attn_weight = False,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_func = rope_func

        self.norm1 = parse_norm_layer_1d(norm_layer, dim)
        self.get_attn_weight = get_attn_weight
        self.attn = AttentionLayer(dim, num_heads, rope_func, get_attn_weight,
                                   False, False, qkv_bias, attn_drop, proj_bias, proj_drop, norm_layer)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = parse_norm_layer_1d(norm_layer, dim)
        self.mlp = Mlp(
            dim, int(dim * mlp_ratio), dim, act_layer=nn.GELU, bias=proj_bias, drop=proj_drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1:torch.Tensor, pos1:torch.Tensor, x2:torch.Tensor, pos2:torch.Tensor):
        x1_after_attn, x1_attn_weight, x2_after_attn, x2_attn_weight = self.attn.forward(
            self.norm1(x1), pos1, self.norm2(x2), pos2
        )
        x1 = x1 + self.drop_path1(x1_after_attn)
        x2 = x2 + self.drop_path2(x2_after_attn)
        x1 = x1 + self.drop_path2(self.mlp(self.norm2(x1)))
        x2 = x2 + self.drop_path2(self.mlp(self.norm2(x2)))
        if self.get_attn_weight:
            return x1, x1_attn_weight, x2, x2_attn_weight
        else:
            return x1, x2
        
    def forward_single(self, x:torch.Tensor, pos:torch.Tensor):
        x_after_attn, x_attn_weight = self.attn.forward(self.norm1(x), pos)
        x = x + self.drop_path1(x_after_attn)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        if self.get_attn_weight:
            return x, x_attn_weight
        else:
            return x
        

class CrossAttnBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: str = 'none',
            rope_func = None,
            get_attn_weight = False,
            reciprocal = True,
        ):
        '''
        self-attn -> cross-attn -> proj  
        when reciprocal: x2 will only function as K, V and it will remain unchanged. And only attn_weight1 is meaningful  
        '''
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_func = rope_func

        self.norm1 = parse_norm_layer_1d(norm_layer, dim)
        self.self_attn = AttentionLayer(dim, num_heads, rope_func, False,
                                        False, False, qkv_bias, attn_drop, proj_bias, proj_drop, norm_layer)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = parse_norm_layer_1d(norm_layer, dim)
        self.get_attn_weight = get_attn_weight
        self.reciprocal = reciprocal
        self.cross_attn = AttentionLayer(dim, num_heads, rope_func, get_attn_weight,
                                   True, reciprocal, qkv_bias, attn_drop, proj_bias, proj_drop, norm_layer)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = parse_norm_layer_1d(norm_layer, dim)
        self.mlp = Mlp(
            dim, int(dim * mlp_ratio), dim, act_layer=nn.GELU, bias=proj_bias, drop=proj_drop
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward_self_attn(self, x1, pos1, x2, pos2):
        if self.reciprocal:
            x1_after_attn, _, x2_after_attn, _ = self.self_attn.forward(
                self.norm1(x1), pos1, self.norm1(x2), pos2, 
            )
            x1 = x1 + self.drop_path1(x1_after_attn)
            x2 = x2 + self.drop_path1(x2_after_attn)
            return x1, x2
        else:
            x1_after_attn, _ = self.self_attn.forward(
                self.norm1(x1), pos1
            )
            x1 = x1 + self.drop_path1(x1_after_attn)
            return x1, x2

    def _forward_cross_attn(self, x1, pos1, x2, pos2):
        x1_after_attn, x1_attn_weight, x2_after_attn, x2_attn_weight = self.cross_attn.forward(
            self.norm2(x1), pos1, self.norm2(x2), pos2
        )
        x1 = x1 + self.drop_path2(x1_after_attn)
        if self.reciprocal:
            x2 = x2 + self.drop_path2(x2_after_attn)
        return x1, x1_attn_weight, x2, x2_attn_weight

    def _forward_proj(self, x1, x2):
        x1 = x1 + self.drop_path3(self.mlp(self.norm3(x1)))
        if self.reciprocal:
            x2 = x2 + self.drop_path3(self.mlp(self.norm3(x2)))
        return x1, x2

    def forward(self, x1:torch.Tensor, pos1:torch.Tensor, x2:torch.Tensor, pos2:torch.Tensor):
        x1, x2 = self._forward_self_attn(x1, pos1, x2, pos2)
        x1, x1_attn_weight, x2, x2_attn_weight = self._forward_cross_attn(
            x1, pos1, x2, pos2
        )
        x1, x2 = self._forward_proj(x1, x2)
        if self.get_attn_weight:
            return x1, x1_attn_weight, x2, x2_attn_weight
        else:
            return x1, x2


class SelfAttnViT(nn.Module):
    def __init__(
            self,
            depth:int,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: str = 'none',
            rope_func = None,
            get_attn_weight = False,
    ):
        super().__init__()
        self.depth = depth
        self.get_attn_weight = get_attn_weight
        self.blocks = nn.ModuleList(
            [
                SelfAttnBlock(dim, num_heads, mlp_ratio, qkv_bias, proj_bias,
                              proj_drop, attn_drop, drop_path, norm_layer, rope_func, get_attn_weight)
                for i in range(depth)
            ]
        )
        self.norm = parse_norm_layer_1d(norm_layer, dim)
    
    def forward(self, x1, pos1, x2, pos2, return_all_blocks=False):
        if return_all_blocks:
            x1_out = []
            x2_out = []
        if self.get_attn_weight:
            attn_weights1 = []
            attn_weights2 = []

        if return_all_blocks:
            if self.get_attn_weight:
                for blk in self.blocks:
                    x1, x1_weight, x2, x2_weight = blk.forward(x1, pos1, x2, pos2)
                    x1_out.append(x1)
                    x2_out.append(x2)
                    attn_weights1.append(x1_weight)
                    attn_weights2.append(x2_weight)
                x1_out[-1] = self.norm(x1_out[-1])
                x2_out[-1] = self.norm(x2_out[-1])
                return x1_out, attn_weights1, x2_out, attn_weights2
            else:
                for blk in self.blocks:
                    x1, x2 = blk.forward(x1, pos1, x2, pos2)
                    x1_out.append(x1)
                    x2_out.append(x2)
                x1_out[-1] = self.norm(x1_out[-1])
                x2_out[-1] = self.norm(x2_out[-1])
                return x1_out, x2_out
        else:
            if self.get_attn_weight:
                for blk in self.blocks:
                    x1, x1_weight, x2, x2_weight = blk.forward(x1, pos1, x2, pos2)
                    attn_weights1.append(x1_weight)
                    attn_weights2.append(x2_weight)
                x1 = self.norm(x1)
                x2 = self.norm(x2)
                return x1, attn_weights1, x2, attn_weights2
            else:
                for blk in self.blocks:
                    x1, x2 = blk.forward(x1, pos1, x2, pos2)
                x1, x2 = self.norm(x1), self.norm(x2)
                return x1, x2
            
    def forward_single(self, x, pos, return_all_blocks=False):
        if return_all_blocks:
            x_out = []
        if self.get_attn_weight:
            attn_weights = []
        if return_all_blocks:
            if self.get_attn_weight:
                for blk in self.blocks:
                    x, x_weight = blk.forward_single(x, pos)
                    x_out.append(x)
                    attn_weights.append(x_weight)
                x_out[-1] = self.norm(x_out[-1])
                return x_out, attn_weights
            else:
                for blk in self.blocks:
                    x = blk.forward_single(x, pos)
                    x_out.append(x)
                x_out[-1] = self.norm(x_out[-1])
                return x_out
        else:
            if self.get_attn_weight:
                for blk in self.blocks:
                    x, x_weight = blk.forward_single(x, pos)
                    attn_weights.append(x_weight)
                x = self.norm(x)
                return x, attn_weights
            else:
                for blk in self.blocks:
                    x = blk.forward_single(x, pos)
                x = self.norm(x)
                return x


class CrossAttnViT(nn.Module):
    def __init__(
            self,
            depth: int,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: str = 'none',
            rope_func = None,
            get_attn_weight = False,
            reciprocal = True,
        ):
        '''
        when reciprocal: x2 will only function as K, V and it will remain unchanged. And only attn_weight1 is meaningful  
        '''
        super().__init__()
        self.depth = depth
        self.get_attn_weight = get_attn_weight
        self.reciprocal = reciprocal
        self.blocks = nn.ModuleList(
            [
                CrossAttnBlock(dim, num_heads, mlp_ratio, qkv_bias, proj_bias, proj_drop, attn_drop, 
                               drop_path, norm_layer, rope_func, get_attn_weight, reciprocal)
                for i in range(depth)
            ]
        )
        self.norm = parse_norm_layer_1d(norm_layer, dim)

    def _forward_reciprocal(self, x1, pos1, x2, pos2, return_all_blocks=False):
        '''same as SelfAttnViT'''
        if return_all_blocks:
            x1_out = []
            x2_out = []
        if self.get_attn_weight:
            attn_weights1 = []
            attn_weights2 = []

        if return_all_blocks:
            if self.get_attn_weight:
                for blk in self.blocks:
                    x1, x1_weight, x2, x2_weight = blk.forward(x1, pos1, x2, pos2)
                    x1_out.append(x1)
                    x2_out.append(x2)
                    attn_weights1.append(x1_weight)
                    attn_weights2.append(x2_weight)
                x1_out[-1] = self.norm(x1_out[-1])
                x2_out[-1] = self.norm(x2_out[-1])
                return x1_out, attn_weights1, x2_out, attn_weights2
            else:
                for blk in self.blocks:
                    x1, x2 = blk.forward(x1, pos1, x2, pos2)
                    x1_out.append(x1)
                    x2_out.append(x2)
                x1_out[-1] = self.norm(x1_out[-1])
                x2_out[-1] = self.norm(x2_out[-1])
                return x1_out, x2_out
        else:
            if self.get_attn_weight:
                for blk in self.blocks:
                    x1, x1_weight, x2, x2_weight = blk.forward(x1, pos1, x2, pos2)
                    attn_weights1.append(x1_weight)
                    attn_weights2.append(x2_weight)
                x1 = self.norm(x1)
                x2 = self.norm(x2)
                return x1, attn_weights1, x2, attn_weights2
            else:
                for blk in self.blocks:
                    x1, x2 = blk.forward(x1, pos1, x2, pos2)
                x1, x2 = self.norm(x1), self.norm(x2)
                return x1, x2

    def _forward_asymmetric(self, x1, pos1, x2, pos2, return_all_blocks=False):
        '''x2 keeps unchanged.'''
        if return_all_blocks:
            x1_out = []
            x2_out = [x2 for _ in range(self.depth)]
        if self.get_attn_weight:
            attn_weights1 = []
            attn_weights2 = None

        if return_all_blocks:
            if self.get_attn_weight:
                for blk in self.blocks:
                    x1, x1_weight, x2, x2_weight = blk.forward(x1, pos1, x2, pos2)
                    x1_out.append(x1)
                    attn_weights1.append(x1_weight)
                x1_out[-1] = self.norm(x1_out[-1])
                return x1_out, attn_weights1, x2_out, attn_weights2
            else:
                for blk in self.blocks:
                    x1, x2 = blk.forward(x1, pos1, x2, pos2)
                    x1_out.append(x1)
                x1_out[-1] = self.norm(x1_out[-1])
                return x1_out, x2_out
        else:
            if self.get_attn_weight:
                for blk in self.blocks:
                    x1, x1_weight, x2, x2_weight = blk.forward(x1, pos1, x2, pos2)
                    attn_weights1.append(x1_weight)
                x1 = self.norm(x1)
                return x1, attn_weights1, x2, attn_weights2
            else:
                for blk in self.blocks:
                    x1, x2 = blk.forward(x1, pos1, x2, pos2)
                x1 = self.norm(x1)
                return x1, x2

    def forward(self, x1, pos1, x2, pos2, return_all_blocks=False):
        if self.reciprocal:
            return self._forward_reciprocal(x1, pos1, x2, pos2, return_all_blocks)
        else:
            return self._forward_asymmetric(x1, pos1, x2, pos2, return_all_blocks)