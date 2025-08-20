import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import parse_norm_layer_1d
from .vga import VGA

# NOTE:
# Although a unified self- and cross-attention implementation that applies attention operation to tokens of 2 images (concatenated into 1 tensor)
# in 1 forward pass may seem general and elegant, we find it to be significantly slower than naively computing attention independently per input image.
# when taking 2 images as input, self-attention is approximately 5.5x slower, and cross-attention approximately 2x slower.  
# This is expected, as performing two separate attention operations incurs linear overhead, whereas doubling sequence length eads to quadratic cost.  
# The main downside of separate attention computing is reduced flexibility. It may be less straightforward to extend to multi-view images (>2 views).


def manual_attn(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, scale,
                drop:float=0., rope_func=None, qpos=None, kpos=None, attn_bias=None, get_attn_weight=False):
    '''
    q, k, v: (B, N, num_heads, head_dim). if unified==True, q,k,v are consist of tokens from 2 images, and the returned attention weight won't distinct the 2 images.  
    qpos, kpos: (B, N, 2)  
    attn_bias: (B, N, N), float tensor. attn_bias will be added to the attention weight before softmax. Typical elements are -inf (dicard) and 0 (preserve)
    return: (B, N, num_heads, head_dim), attn_weight: (B, H, N, N)  
    '''
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)  # B,num_head, N, head_dim
    if rope_func is not None and qpos is not None:
        q = rope_func(q, qpos)
        k = rope_func(k, qpos if kpos is None else kpos)
    attn_before_softmax = (q * scale @ k.transpose(-1, -2))  # (B,num_heads,N,head_dim) @ (B,num_heads,head_dim,N) -> (B,num_heads,N,N)
    attn_softmax = F.softmax(attn_before_softmax + attn_bias if attn_bias is not None else attn_before_softmax, dim=-1)
    if drop is not None and drop > 0.:
        attn_softmax = F.dropout(attn_softmax, drop)
    x = (attn_softmax @ v).transpose(1,2).contiguous()  # B,N,num_heads,head_dim
    if get_attn_weight:
        return x, attn_before_softmax
    else:
        return x
    
def manual_attn_unified(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, scale, unified:bool,
                cross_attn=False, drop:float=None, rope_func=None, qpos=None, kpos=None, get_attn_weight=False):
    '''
    WARNING: The correctness of the unified self- and cross-attention implementation has not been tested!  

    q, k, v: B, N, num_heads, head_dim. if unified==True, q,k,v are consist of tokens from 2 images, and the returned attention weight won't distinct the 2 images.  
    qpos, kpos: (B, N, 2)  
    return: B, N, num_heads, head_dim  
    '''
    B, N, H, D = q.shape
    attn_bias = None
    if unified:
        if qpos is not None and qpos.shape[1] != N:
            assert qpos.shape[1] == N // 2
            qpos = torch.cat((qpos, qpos), dim=1)
        if kpos is not None and kpos.shape[1] != N:
            assert kpos.shape[1] == N // 2
            kpos = torch.cat((kpos, kpos), dim=1)
        attn_bias = torch.full((B, H, N, N), -10000.0, dtype=q.dtype, device=q.device)
        if cross_attn:
            attn_bias[..., :N//2, N//2:] = 0
            attn_bias[..., N//2:, :N//2] = 0
        else:
            attn_bias[..., :N//2, :N//2] = 0
            attn_bias[..., N//2:, N//2:] = 0

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)  # B,num_head, N, head_dim

    if rope_func is not None and qpos is not None:
        q = rope_func(q, qpos)
        k = rope_func(k, qpos if kpos is None else kpos)
    attn_before_softmax = (q @ k.transpose(-1, -2)) * scale  # (B,num_heads,N,head_dim) @ (B,num_heads,head_dim,N) -> (B,num_heads,N,N)
    # print(attn_before_softmax.shape, attn_bias.shape if attn_bias is not None else "None")
    
    attn_softmax = F.softmax(attn_before_softmax + attn_bias if attn_bias is not None else attn_before_softmax, dim=-1)
    if drop is not None:
        attn_softmax = F.dropout(attn_softmax, drop)
    x = (attn_softmax @ v).transpose(1,2).contiguous()  # B,N,num_heads,head_dim
    if get_attn_weight:
        return x, attn_before_softmax
    else:
        return x
    
try:
    import xformers.ops as xops
    mem_efficient_attn_operation = xops.memory_efficient_attention
except ImportError as e:
    print("xformers has not been installed. Using torch.nn.functional.scaled_dot_product_attention instead.")
    def mem_efficient_attn_operation(q, k, v, attn_bias, p=0, scale=None):
        # q, k, v: (B, N, H, D)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        return F.scaled_dot_product_attention(q, k, v, attn_bias, p, scale=scale).transpose(1,2)
    
def mem_efficient_attn(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, scale,
                    drop:float=0., rope_func=None, qpos=None, kpos=None, attn_bias=None, get_attn_weight=False):
    '''
    q, k, v: (B, N, num_heads, head_dim). if unified==True, q,k,v are consist of tokens from 2 images, and the returned attention weight won't distinct the 2 images.  
    qpos, kpos: (B, N, 2)  
    attn_bias: (B, N, N), float tensor. attn_bias will be added to the attention weight before softmax. Typical elements are -inf (dicard) and 0 (preserve)  
    attn_bias can also be xformers.ops.AttentionBias (when xformers has been installed). See https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.BlockDiagonalMask for its usage.
    return: B, N, num_heads, head_dim  
    '''
    if rope_func is not None and qpos is not None:
        q = rope_func(q.transpose(1,2), qpos).transpose(1,2)  # (B,N,H,D)
        k = rope_func(k.transpose(1,2), qpos if kpos is None else kpos).transpose(1,2)

    x = mem_efficient_attn_operation(
        q, k, v, attn_bias, p=drop, scale=scale
    )  # B, H, N, D
    return x
    

def mem_efficient_attn_unified(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, scale, unified,
                    cross_attn=False, drop:float=None, rope_func=None, qpos=None, kpos=None, get_attn_weight=False):
    '''
    WARNING: The correctness of the unified self- and cross-attention implementation has not been tested!  
    '''
    B, N, H, D = q.shape
    attn_bias = None
    if unified:
        if qpos is not None and qpos.shape[1] != N:
            assert qpos.shape[1] == N // 2
            qpos = torch.cat((qpos, qpos), dim=1)
        if kpos is not None and kpos.shape[1] != N:
            assert kpos.shape[1] == N // 2
            kpos = torch.cat((kpos, kpos), dim=1)
        if cross_attn:
            attn_bias = torch.full((B, H, N, N), -10000.0, dtype=q.dtype, device=q.device)
            attn_bias[..., :N//2, N//2:] = 0
            attn_bias[..., N//2:, :N//2] = 0
        else:
            seqlen = N // 2
            attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([seqlen] * (B*2), device=q.device)
            # attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([N//2, N//2], device=q.device)

    if rope_func is not None and qpos is not None:
        q = rope_func(q.transpose(1,2), qpos).transpose(1,2)
        k = rope_func(k.transpose(1,2), qpos if kpos is None else kpos).transpose(1,2)

    if unified and not cross_attn:
        q = q.view(1, -1, H, D)
        k = k.view(1, -1, H, D)
        v = v.view(1, -1, H, D)

    x = mem_efficient_attn_operation(
        q, k, v, attn_bias, p=drop, scale=scale
    )  # B, N, H, D

    if unified and not cross_attn:
        x = x.view(B, N, H, D)

    return x # B,N,H,D


def mem_efficient_attn_unified_rotateQ(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, scale, unified,
                    cross_attn=False, drop:nn.Module=None, rope_func=None, qpos=None, kpos=None, get_attn_weight=False):
    '''
    WARNING: The correctness of the unified self- and cross-attention implementation has not been tested!  

    Cross-attention by swaping the tokens of 2 input images is faster than using a custom Tensor attn_bias (about 18%, 0.002062s vs 0.002517s)  
    '''
    B, N, H, D = q.shape
    attn_bias = None
    unified_and_cross = (unified and cross_attn)
    if unified:
        seq_len = N // 2
        # attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([N//2, N//2], device=q.device)
        attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([seq_len] * (B*2), device=q.device)
        if qpos is not None and qpos.shape[1] != N:
            assert qpos.shape[1] == N // 2
            qpos = torch.cat((qpos, qpos), dim=1)
        if kpos is not None and kpos.shape[1] != N:
            assert kpos.shape[1] == N // 2
            kpos = torch.cat((kpos, kpos), dim=1)

    if rope_func is not None and qpos is not None:
        q = rope_func(q.transpose(1,2), qpos).transpose(1,2)
        k = rope_func(k.transpose(1,2), qpos if kpos is None else kpos).transpose(1,2)

    if unified_and_cross:
        q = torch.cat([q[:, N//2:], q[:, :N//2]], dim=1).contiguous()

    if unified:
        q = q.view(1, -1, H, D)
        k = k.view(1, -1, H, D)
        v = v.view(1, -1, H, D)

    x = mem_efficient_attn_operation(
        q, k, v, attn_bias, p=drop, scale=scale
    )  # B,N,H,D

    if unified:
        x = x.view(B, N, H, D)

    if unified_and_cross:
        x = torch.cat([x[:, N//2:], x[:,:N//2]], dim=1).contiguous()

    return x


class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=12, rope_func=None, get_attn_weight=False, cross_attn=False, reciprocal=False,
                 qkv_bias=True, attn_drop=0., proj_bias=True, proj_drop=0., norm_layer=None,
                 # gate
                 gate=False, gate_type='cond_per_head', vga=True, gate_mlp_ratio=0.25, gate_by_all_feat=False):
        '''
        cross_attn: if cross_attn, it is specified to be a cross-attention layer, and tokens of 2 input images should be inputted in forward pass.  
        reciprocal: it only takes effect when cross_attn==True. if reciprocal, it will calculate cross-attention alternatively on image1 and image2.
        '''
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.rope_func = rope_func
        self.cross_attn = cross_attn
        self.reciprocal = reciprocal
        self.get_attn_weight = get_attn_weight
        self.attn_func = mem_efficient_attn if not get_attn_weight else manual_attn
        self.attn_drop = attn_drop

        # self.use_qkv_layer = (not self.cross_attn) or self.reciprocal
        self.use_qkv_layer = (not self.cross_attn)  
        # CuRoPE is an inplace operation. If it's a symmetric cross-attention layer (i.e., it needs to compute both cross_attn(q1,k2,v2) and cross_attn(q2,k1,v1)
        # simultaneously), using a single QKV linear layer without splitting it will cause the backward pass to fail due to the inplace operation!!

        if self.use_qkv_layer:
            self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        else:
            self.projq = nn.Linear(dim, dim, bias=qkv_bias)
            self.projk = nn.Linear(dim, dim, bias=qkv_bias)
            self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_norm = parse_norm_layer_1d(norm_layer, dim)

        # gate
        self.gate = gate
        self.vga = vga
        self.vga_module = VGA(
            self.dim, self.num_heads, gate_type, gate_mlp_ratio, gate_by_all_feat=gate_by_all_feat
        )


    def _x_to_qkv(self, x, q:bool=True, k:bool=True, v:bool=True):
        '''When self.use_qkv_layer is False, q, k and v are used to specify whther this x is Q, K or V.'''
        B,N,C = x.shape
        if self.use_qkv_layer:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,1,3,4).contiguous() # 3,B,N,H,D
            # q, k, v = qkv.unbind(0)  # it will Cause curope to crash 
            q, k, v = [qkv[i] for i in range(3)]
        else:
            q = self.projq(x).reshape(B, N, self.num_heads, C // self.num_heads) if q else None
            k = self.projk(x).reshape(B, N, self.num_heads, C // self.num_heads) if k else None
            v = self.projv(x).reshape(B, N, self.num_heads, C // self.num_heads) if v else None
        return q, k, v
    
    def _post_attn_proj(self, x):
        B, N = x.shape[:2]
        x = x.view(B, N, -1)
        return self.proj_drop(self.proj(self.proj_norm(x)))

    def forward(self, x1:torch.Tensor, pos1:torch.Tensor, x2:torch.Tensor=None, pos2:torch.Tensor=None):
        '''
        return: x1, attn_weight1, (x2, attn_weight2). if self.get_attn_weight==False, attn_weight is None.  
        if self.cross_attn==True and self.reprocal==False, only x1, attn_weight1 is meaningful.  
        '''
        B, N, C = x1.shape
        attn_drop = self.attn_drop if self.training else 0.
        x2_input = x2 is not None

        x1_origin = x1
        x2_origin = x2

        if not self.cross_attn:
            q1, k1, v1 = self._x_to_qkv(x1)
            if x2 is not None:
                q2, k2, v2 = self._x_to_qkv(x2)
            else:
                q2 = k2 = v2 = None
            attn_weight1 = None

            x1 = self.attn_func(
                q1, k1, v1, self.scale, attn_drop, self.rope_func, pos1, None, None, self.get_attn_weight
            )
            if self.get_attn_weight:
                x1, attn_weight1 = x1
            # gate
            if self.gate:
                x1 = self.vga_module.forward(v1 if self.vga else x1_origin, x1)
            
            # porjection after attn
            x1 = self._post_attn_proj(x1)
            if q2 is not None:
                attn_weight2 = None
                x2 = self.attn_func(
                    q2, k2, v2, self.scale, attn_drop, self.rope_func, pos2, None, None, self.get_attn_weight
                )
                if self.get_attn_weight:
                    x2, attn_weight2 = x2
                # gate
                if self.gate:
                    x2 = self.vga_module.forward(v2 if self.vga else x2_origin, x2)
                # projection after attn
                x2 = self._post_attn_proj(x2)
        else:
            if self.reciprocal:
                q1, k1, v1 = self._x_to_qkv(x1)
                q2, k2, v2 = self._x_to_qkv(x2)
            else:
                q1, _, _ = self._x_to_qkv(x1, True, False, False)
                _, k2, v2 = self._x_to_qkv(x2, False, True, True)
            attn_weight1 = attn_weight2 = None
            x1 = self.attn_func(
                q1, k2, v2, self.scale, attn_drop, self.rope_func, pos1, pos2, None, self.get_attn_weight
            )
            if self.get_attn_weight:
                x1, attn_weight1 = x1
            # gate
            if self.gate:
                x1 = self.vga_module.forward(v2 if self.vga else x1_origin, x1)
            # projection after attn
            x1 = self._post_attn_proj(x1)
            if self.reciprocal:
                x2 = self.attn_func(
                    q2, k1, v1, self.scale, attn_drop, self.rope_func, pos2, pos1, None, self.get_attn_weight
                )
                if self.get_attn_weight:
                    x2, attn_weight2 = x2
                # gate
                if self.gate:
                    x2 = self.vga_module.forward(v1 if self.vga else x2_origin, x2)
                # projection after attn
                x2 = self._post_attn_proj(x2)
        if not x2_input:
            return x1, attn_weight1
        else:
            return x1, attn_weight1, x2, attn_weight2
    


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from models.curope import cuRoPE2D as RoPE2D
    from models.blocks import PositionGetter

    test_time = False
    test_correctness = True
    bf16 = True

    warmup = 100
    run = 100
    device = torch.device("cuda")

    h, w = 224, 224
    patch_size = 16
    D = 768
    H = 12
    HD = D // H
    scale = HD ** -0.5
    B = 64
    N = (h // patch_size) * (w // patch_size)

    pos_getter = PositionGetter()
    rope = RoPE2D()

    img1_tokens = torch.rand(B, N, H, HD, dtype=torch.float32, device=device)
    img2_tokens = torch.rand(B, N, H, HD, dtype=torch.float32, device=device)

    pos = pos_getter(B, h // patch_size, w // patch_size, img1_tokens.device)

    ###################################################################
    print("auto cast bf16: ", bf16)
    with torch.autocast('cuda', enabled=bf16):
        if test_time:
            print("manual attention, separate self-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                start = time.time()
                x1 = manual_attn_unified(img1_tokens, img1_tokens, img1_tokens, scale, False, False, 0., rope, pos, pos, False)
                x2 = manual_attn_unified(img2_tokens, img2_tokens, img2_tokens, scale, False, False, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

            print("manual attention, separate cross-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                start = time.time()
                x1 = manual_attn_unified(img1_tokens, img2_tokens, img2_tokens, scale, False, True, 0., rope, pos, pos, False)
                x2 = manual_attn_unified(img2_tokens, img1_tokens, img1_tokens, scale, False, True, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

            print("manual attention, union self-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                qkv = torch.cat([img1_tokens, img2_tokens], dim=1).contiguous()
                start = time.time()
                x1 = manual_attn_unified(qkv, qkv, qkv, scale, True, False, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

            print("manual attention, union cross-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                qkv = torch.cat([img1_tokens, img2_tokens], dim=1).contiguous()
                start = time.time()
                x1 = manual_attn_unified(qkv, qkv, qkv, scale, True, True, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

            print("xformers attention, separate self-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                start = time.time()
                x1 = mem_efficient_attn_unified(img1_tokens, img1_tokens, img1_tokens, scale, False, False, 0., rope, pos, pos, False)
                x2 = mem_efficient_attn_unified(img2_tokens, img2_tokens, img2_tokens, scale, False, False, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

            print("xformers attention, separate cross-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                start = time.time()
                x1 = mem_efficient_attn_unified(img1_tokens, img2_tokens, img2_tokens, scale, False, True, 0., rope, pos, pos, False)
                x2 = mem_efficient_attn_unified(img2_tokens, img1_tokens, img1_tokens, scale, False, True, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

            print("xformers attention, union self-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                qkv = torch.cat([img1_tokens, img2_tokens], dim=1).contiguous()
                start = time.time()
                x1 = mem_efficient_attn_unified(qkv, qkv, qkv, scale, True, False, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

            print("xformers attention, union cross-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                qkv = torch.cat([img1_tokens, img2_tokens], dim=1).contiguous()
                start = time.time()
                x1 = mem_efficient_attn_unified(qkv, qkv, qkv, scale, True, True, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

            print("xformers attention, union rotate Q cross-attention.")
            cnt = 0
            t = 0
            for i in tqdm(range(warmup + run)):
                qkv = torch.cat([img1_tokens, img2_tokens], dim=1).contiguous()
                start = time.time()
                x1 = mem_efficient_attn_unified_rotateQ(qkv, qkv, qkv, scale, True, True, 0., rope, pos, pos, False)
                if i >= warmup:
                    t += (time.time() - start)
                    cnt += 1
            print("time: ", t / cnt)
            print(x1.shape)

        ###################################################################
        if test_correctness:
            # with torch.autocast('cuda', torch.bfloat16):
                # NOTE: cuRoPE is a inplace operation!
                if bf16:
                    img1_tokens = img1_tokens.to(torch.bfloat16)
                    img2_tokens = img2_tokens.to(torch.bfloat16)

                img1_origin = img1_tokens.clone()
                img2_origin = img2_tokens.clone()
                manual_x1_self = manual_attn(img1_tokens, img1_tokens, img1_tokens, scale, 0, rope, pos, pos, None, False)
                manual_x2_self = manual_attn(img2_tokens, img2_tokens, img2_tokens, scale, 0, rope, pos, pos, None, False)
                img1_tokens = img1_origin.clone()
                img2_tokens = img2_origin.clone()
                xformer_x1_self = mem_efficient_attn(img1_tokens, img1_tokens, img1_tokens, scale, 0, rope, pos, pos, None, False)
                img1_tokens = img1_origin.clone()
                img2_tokens = img2_origin.clone()
                xformer_x2_self = mem_efficient_attn(img2_tokens, img2_tokens, img2_tokens, scale, 0, rope, pos, pos, None, False)
                img1_tokens = img1_origin.clone()
                img2_tokens = img2_origin.clone()

                manual_x1_cross = manual_attn(img1_tokens, img2_tokens, img2_tokens, scale, 0, rope, pos, pos, None, False)
                img1_tokens = img1_origin.clone()
                img2_tokens = img2_origin.clone()
                manual_x2_cross = manual_attn(img2_tokens, img1_tokens, img1_tokens, scale, 0, rope, pos, pos, None, False)
                img1_tokens = img1_origin.clone()
                img2_tokens = img2_origin.clone()
                xformer_x1_cross = mem_efficient_attn(img1_tokens, img2_tokens, img2_tokens, scale, 0, rope, pos, pos, None, False)
                img1_tokens = img1_origin.clone()
                img2_tokens = img2_origin.clone()
                xformer_x2_cross = mem_efficient_attn(img2_tokens, img1_tokens, img1_tokens, scale, 0, rope, pos, pos, None, False)
                
                print(f"manual_x1_self - xformer_x1_self: {torch.mean(torch.abs(manual_x1_self - xformer_x1_self))}")
                print(f"manual_x2_self - xformer_x2_self: {torch.mean(torch.abs(manual_x2_self - xformer_x2_self))}")
                print(f"manual_x1_cross - xformer_x1_cross: {torch.mean(torch.abs(manual_x1_cross - xformer_x1_cross))}")
                print(f"manual_x2_cross - xformer_x2_cross: {torch.mean(torch.abs(manual_x2_cross - xformer_x2_cross))}")