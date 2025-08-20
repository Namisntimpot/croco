import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.blocks import SelfAttnViT, CrossAttnViT
from .layers.patch_embd import PositionGetter, PatchEmbed
from .layers.utils import to_2tuple, patchify, unpatchify
from .masking import RandomMask
from .pos_embed import RoPE2D


def merge_attn_map(attn_maps, suppress_1st_attn=False, attn_layers_adopted:list[int]=None):
    '''attn_maps: list of (B, num_heads, N, N)'''
    attn_maps = attn_maps if attn_layers_adopted is None else [attn_maps[idx] for idx in attn_layers_adopted]
    attn_maps = torch.stack(attn_maps, dim=1)  # B, n_blk, n_heads, N, N. dim=-2: tokens of img1  
    attn_maps = torch.mean(attn_maps, dim=(1,2)) # B, N, N
    if suppress_1st_attn:
        attn_maps[:,:,0] = attn_maps.min()  # reference to ZeroCo
    return attn_maps

def weighted_sum_attn_map(corr:torch.Tensor, beta=0.02):
    r'''
    SFNet: Learning Object-aware Semantic Flow (Lee et al.)  
    corr: B*(img_size//patch_size)*(img_size//patch_size) * num_patches, the (img_size//patch_size)*(img_size//patch_size) indicates source image tokens.
    beta: temperature.  
    '''
    b,h,w,p = corr.shape
    corr = F.softmax(corr / beta, dim=-1)
    corr = corr.view(-1, h, w, h, w)  # (source hxw) x (target hxw)  

    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=corr.device), torch.arange(w, device=corr.device), indexing='ij')  # hxw
    grid = torch.stack([grid_x, grid_y], dim=0).float()  # grid coordinate, 2xhxw
    corresp = torch.sum(corr.unsqueeze(-3) * grid, dim=(-2,-1)).permute(0, 3, 1, 2)  # b*2*source_h*source_w
    flow = corresp - grid

    return corresp, flow


class MMAEViT(nn.Module):
    def __init__(
            self,
            img_size:int = 224,
            patch_size:int = 16,
            mask_ratio:float = 0.9,
            enc_depth_self_attn:int = 8,
            enc_depth_cros_attn:int = 8,
            enc_dim: int = 768,
            enc_num_heads: int = 16,
            dec_depth:int = 8,
            dec_dim: int = 512,
            dec_num_heads: int = 16,
            mlp_ratio:int = 4,
            norm_layer:str = 'layer_norm',
            pos_embd:str = 'RoPE100',  # here we only support RoPE, deprecating cosine, as RoPE performs much better than cosine
            get_attn_weight: bool = False,
            # REG
            num_reg_tokens = 0,
            dec_with_reg = False,  # deprecate register tokens before decoding.
            # VGA
            gate = False,
            gate_type = 'cond_per_head',
            vga = True,
            gate_mlp_ratio = 0.25,
            gate_by_all_feat = False,
        ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)

        # patch embeddings  (with initialization done as in MAE)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_dim)

        # mask generations
        self.mask_generator = RandomMask(self.patch_embed.num_patches, mask_ratio)

        # positional embedding
        assert pos_embd != "cosine", "We only support RoPE, deprecating cosine, as RoPE performs much better than cosine"
        self.pos_embd = pos_embd
        if pos_embd.lower().startswith("rope"):
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            base = float(pos_embd[len('RoPE'):])
            self.rope = RoPE2D(freq=base)
        else:
            raise NotImplementedError(f"not implemeneted pos_embd: {pos_embd}")
        
        self.get_attn_weight = get_attn_weight
        # self-attention encoder
        self.self_attn_encoder = SelfAttnViT(
            enc_depth_self_attn, enc_dim, enc_num_heads, mlp_ratio, qkv_bias=True,
            proj_bias=True, norm_layer=norm_layer, rope_func=self.rope if hasattr(self, 'rope') else None, get_attn_weight=False,
            gate=gate, gate_type=gate_type, vga=vga, gate_mlp_ratio=gate_mlp_ratio, gate_by_all_feat=gate_by_all_feat
        )
        # cross-attention encoder
        self.cros_attn_encoder = self._set_cross_attn_encoder(
            enc_depth_cros_attn, enc_dim, enc_num_heads, mlp_ratio, norm_layer, 
            self.rope if hasattr(self, 'rope') else None, get_attn_weight=False
        )

        # self-attention decoder
        self.decoder_embed = nn.Linear(enc_dim, dec_dim, bias=True)
        self.self_attn_decoder = SelfAttnViT(
            dec_depth, dec_dim, dec_num_heads, mlp_ratio, qkv_bias=True, proj_bias=True,
            norm_layer=norm_layer, rope_func=self.rope if hasattr(self, 'rope') else None, get_attn_weight=False,
            gate=gate, gate_type=gate_type, vga=vga, gate_mlp_ratio=gate_mlp_ratio, gate_by_all_feat=gate_by_all_feat
        )

        # mask_token.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        # register_token
        self.num_reg_tokens = num_reg_tokens
        self.dec_with_reg = dec_with_reg
        if self.num_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg_tokens, enc_dim))

        # prediction_head
        self.in_channels = 3
        self.prediction_head = nn.Linear(dec_dim, patch_size**2 * self.in_channels, bias=True)

        self.initialize_weights()


    def _set_cross_attn_encoder(self, enc_depth_cros_attn, 
                                enc_dim, enc_num_heads, mlp_ratio, 
                                norm_layer, rope, get_attn_weight,
                                gate, gate_type, vga, gate_mlp_ratio, gate_by_all_feat):
        # for overriding.
        return CrossAttnViT(
            enc_depth_cros_attn, enc_dim, enc_num_heads, mlp_ratio, qkv_bias=True, proj_bias=True,
            norm_layer=norm_layer, rope_func=rope, get_attn_weight=get_attn_weight, reciprocal=True,
            gate=gate, gate_type=gate_type, vga=vga, gate_mlp_ratio=gate_mlp_ratio, gate_by_all_feat=gate_by_all_feat
        )

    
    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        self.patch_embed._init_weights()

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)
        if self.num_reg_tokens > 0:
            torch.nn.init.normal_(self.reg_tokens, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cat_register_tokens(self, reg_tokens, x, pos, posvis, mask):
        b, nreg = reg_tokens.shape[:2]
        reg_pos = PositionGetter.zero_postitions(b, nreg, x.device)
        reg_mask = torch.zeros((b, nreg), dtype=bool)
        x = torch.cat((reg_tokens, x), dim=1).contiguous()
        pos = torch.cat((reg_pos, pos), dim=1).contiguous()
        posvis = torch.cat((reg_pos, posvis), dim=1).contiguous()
        mask = torch.cat((reg_mask, mask), dim=1).contiguous()
        return x, pos, posvis, mask
    
    def exclude_register_tokens(self, x, pos, mask, attn_weight=None):
        x = x[:, self.num_reg_tokens:]
        pos = pos[:, self.num_reg_tokens:]
        mask = mask[:, self.num_reg_tokens:]
        if attn_weight is not None:
            # attn_weight: B, N, N
            attn_weight = attn_weight[:, self.num_reg_tokens:, self.num_reg_tokens:]
        return x, pos, mask, attn_weight

    def _encode_image(self, image1:torch.Tensor, image2:torch.Tensor, do_mask1=False, do_mask2=False, return_all_blocks = False):
        """
        image has B x 3 x img_size x img_size 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """
        x1, pos1, posvis1, masks1, x2, pos2, posvis2, masks2 = self._encode_image_self_attn_stage(
            image1, image2, do_mask1, do_mask2, return_all_blocks
        )
        if return_all_blocks:
            out1, out2 = x1, x2
            x1, x2 = x1[-1], x2[-1]

        if self.get_attn_weight:
            x1, attn_weight1, x2, attn_weight2 = self._encode_image_cross_attn_stage(
                x1, posvis1, x2, posvis2, return_all_blocks
            )
        else:
            x1, x2 = self._encode_image_cross_attn_stage(
                x1, posvis1, x2, posvis2, return_all_blocks
            )
        if return_all_blocks:
            out1 += x1
            out2 += x2
        else:
            out1, out2 = x1, x2

        if self.get_attn_weight:
            return out1, attn_weight1, pos1, masks1, out2, attn_weight2, pos2, masks2
        else:
            return out1, pos1, masks1, out2, pos2, masks2
        

    def _encode_image_self_attn_stage(self, image1:torch.Tensor, image2:torch.Tensor, do_mask1=False, do_mask2=False, return_all_blocks = False):
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x1, pos1 = self.patch_embed(image1)
        x2, pos2 = self.patch_embed(image2)

        # apply masking 
        B,N,C = x1.size()
        if do_mask1:
            masks1 = self.mask_generator(x1)  # B,N
            x1 = x1[~masks1].view(B, -1, C)
            posvis1 = pos1[~masks1].view(B, -1, 2).contiguous()
        else:
            B,N,C = x1.size()
            masks1 = torch.zeros((B,N), dtype=bool)
            posvis1 = pos1
        if do_mask2:
            masks2 = self.mask_generator(x2)
            x2 = x2[~masks2].view(B, -1, C)
            posvis2 = pos2[~masks2].view(B, -1, 2).contiguous()
        else:
            B,N,C = x2.size()
            masks2 = torch.zeros((B, N), dtype=bool)
            posvis2 = pos2

        # add register tokens
        if self.num_reg_tokens > 0:
            reg_tokens = self.reg_tokens.expand(B, self.num_reg_tokens, -1)
            x1, pos1, posvis1, masks1 = self.cat_register_tokens(
                reg_tokens, x1, pos1, posvis1, masks1
            )
            x2, pos2, posvis2, masks2 = self.cat_register_tokens(
                reg_tokens, x2, pos2, posvis2, masks2
            )
        
        # now apply the transformer encoder and normalization   
        # self attention encoder
        x1, x2 = self.self_attn_encoder.forward(x1, posvis1, x2, posvis2, return_all_blocks) # no attention weight.
        return x1, pos1, posvis1, masks1, x2, pos2, posvis2, masks2
    
    def _encode_image_cross_attn_stage(self, x1:torch.Tensor, posvis1, x2:torch.Tensor, posvis2, return_all_blocks=False):
        # cross attention encoder
        if self.get_attn_weight:
            x1, attn_weight1, x2, attn_weight2 = self.cros_attn_encoder.forward(
                x1, posvis1, x2, posvis2, return_all_blocks
            )
        else:
            x1, x2 = self.cros_attn_encoder.forward(
                x1, posvis1, x2, posvis2, return_all_blocks
            )
        if self.get_attn_weight:
            return x1, attn_weight1, x2, attn_weight2
        else:
            return x1, x2
        

    def _decode(self, feat1, pos1, masks1, feat2, pos2, masks2, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
                           
        masks1 can be None => assume image1 fully visible 
        """
        # encoder to decoder layer 
        visf1 = self.decoder_embed(feat1)
        visf2 = self.decoder_embed(feat2)

        # append masked tokens to the sequence
        B, Nenc, C = visf1.shape
        if masks1 is None:  # eval, downstream
            f1_ = visf1
        else:               # pretraining
            Ntotal = masks1.shape[1]  # B,N
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B*Nenc, C)
        if masks2 is None:
            f2_ = visf2
        else:
            Ntotal = masks2.shape[1]
            f2_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf2.dtype)
            f2_[~masks2] = visf2.view(B*Nenc, C)
        
        f1_, f2_ = self.self_attn_decoder.forward(
            f1_, pos1, f2_, pos2, return_all_blocks
        )
        
        return f1_, f2_
    

    def forward(self, img1, img2, do_mask=False, **kwargs):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size
        do_mask: mask both img1 and img2.  
        kwarges contains inference argument for attention weight-based matching task: 
          reciprocity=True, suppress_1st_token=False, attn_layers_adopted:list[int]=None, temperature=0.02
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case 
        """
        # encoder
        if self.get_attn_weight:
            feat1, attn_weight1, pos1, masks1, feat2, attn_weight2, pos2, masks2 = self._encode_image(
                img1, img2, do_mask, do_mask, False
            )
        else:
            feat1, pos1, masks1, feat2, pos2, masks2 = self._encode_image(
                img1, img2, do_mask, do_mask, False
            )
        # decoder
        if self.num_reg_tokens > 0 and not self.dec_with_reg:
            feat1, pos1, masks1, attn_weight1 = self.exclude_register_tokens(
                feat1, pos1, masks1, attn_weight1 if self.get_attn_weight else None
            )
            feat2, pos2, masks2, attn_weight2 = self.exclude_register_tokens(
                feat2, pos2, masks2, attn_weight2 if self.get_attn_weight else None
            )

        decfeat1, decfeat2 = self._decode(feat1, pos1, masks1, feat2, pos2, masks2, False)

        if self.num_reg_tokens > 0 and self.dec_with_reg:
            decfeat1, pos1, masks1, attn_weight1 = self.exclude_register_tokens(
                decfeat1, pos1, masks1, attn_weight1 if self.get_attn_weight else None
            )
            decfeat2, pos2, masks2, attn_weight2 = self.exclude_register_tokens(
                decfeat2, pos2, masks2, attn_weight2 if self.get_attn_weight else None
            )

        # prediction head
        out1 = self.prediction_head(decfeat1)
        out2 = self.prediction_head(decfeat2)

        # get target
        target1 = patchify(img1, self.patch_size)
        target2 = patchify(img2, self.patch_size)

        if not self.get_attn_weight:
            return out1, masks1, target1, out2, masks2, target2
        else:
            attn_weight, corresp, flow = self.process_attn_weight(attn_weight1, attn_weight2, **kwargs)
            return out1, masks1, target1, out2, masks2, target2, attn_weight, corresp, flow
    

    def process_attn_weight(self, weight1:list, weight2:list,
                            reciprocity=True, suppress_1st_token=False, attn_layers_adopted:list[int]=None, temperature=0.02, **kwargs):
        '''
        weight1 & weight2: list of B*num_heads*N*N, dim=-2: tokens of "Q".  
        weight1: take img1 as Q, img2 as K, V.  
        weight2: take img2 as Q, img1 as K, V.
        '''
        weight1 = merge_attn_map(weight1, suppress_1st_token, attn_layers_adopted)
        if reciprocity:
            weight2 = merge_attn_map(weight2, suppress_1st_token, attn_layers_adopted)
            weight = (weight1 + weight2.transpose(-1, -2)) / 2
        else:
            weight = weight1

        corresp, flow = weighted_sum_attn_map(
            weight.view(weight.shape[0], self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1], -1), temperature
        )

        return weight, corresp, flow
    

class CroCoViT(MMAEViT):
    def __init__(
            self,
            img_size:int = 224,
            patch_size:int = 16,
            mask_ratio:float = 0.9,
            enc_depth_self_attn:int = 8,
            enc_depth_cros_attn:int = 8,
            enc_dim: int = 768,
            enc_num_heads: int = 16,
            dec_depth:int = 8,
            dec_dim: int = 512,
            dec_num_heads: int = 16,
            mlp_ratio:int = 4,
            norm_layer:str = 'layer_norm',
            pos_embd:str = 'RoPE100',  # here we only support RoPE, deprecating cosine, as RoPE performs much better than cosine
            get_attn_weight: bool = False,
            # VGA
            gate = False,
            gate_type = 'cond_per_head',
            vga = True,
            gate_mlp_ratio = 0.25,
            gate_by_all_feat = False,
        ):
        super().__init__(img_size, patch_size, mask_ratio, enc_depth_self_attn, 
                         enc_depth_cros_attn, enc_dim, enc_num_heads, dec_depth, dec_dim, 
                         dec_num_heads, mlp_ratio, norm_layer, pos_embd, get_attn_weight,
                         gate, gate_type, vga, gate_mlp_ratio, gate_by_all_feat)
        
    def _set_cross_attn_encoder(self, enc_depth_cros_attn, 
                                enc_dim, enc_num_heads, mlp_ratio, 
                                norm_layer, rope, get_attn_weight,
                                gate, gate_type, vga, gate_mlp_ratio, gate_by_all_feat):
        return CrossAttnViT(
            enc_depth_cros_attn, enc_dim, enc_num_heads, mlp_ratio, qkv_bias=True, proj_bias=True,
            norm_layer=norm_layer, rope_func=rope, get_attn_weight=get_attn_weight, reciprocal=False,
            gate=gate, gate_type=gate_type, vga=vga, gate_mlp_ratio=gate_mlp_ratio, gate_by_all_feat=gate_by_all_feat
        )
    

    def _decode(self, feat1, pos1, masks1, return_all_blocks=False):
        visf1 = self.decoder_embed(feat1)
        
        B, Nenc, C = visf1.shape
        if masks1 is None:  # eval, downstream
            f1_ = visf1
        else:               # pretraining
            Ntotal = masks1.shape[1]  # B,N
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B*Nenc, C)

        f1_ = self.self_attn_decoder.forward_single(
            f1_, pos1, return_all_blocks
        )
        return f1_
    

    def _encode_image(self, image1, image2, do_mask1=False, do_mask2=False, return_all_blocks=False, reciprocal=True):
        if not self.get_attn_weight or not reciprocal:
            return super()._encode_image(image1, image2, do_mask1, do_mask2, return_all_blocks)
        
        # get_attn_weight and reciprocal.
        x1, pos1, posvis1, masks1, x2, pos2, posvis2, masks2 = self._encode_image_self_attn_stage(
            image1, image2, do_mask1, do_mask2, return_all_blocks
        )
        if return_all_blocks:
            out1, out2 = x1, x2
            x1, x2 = x1[-1], x2[-1]

        x1_, attn_weight1, _, _ = self._encode_image_cross_attn_stage(
            x1, posvis1, x2, posvis2, return_all_blocks
        )
        x2_, attn_weight2, _, _ = self._encode_image_cross_attn_stage(
            x2, posvis2, x1, posvis1, return_all_blocks=False
        )
        x1 = x1_
        if return_all_blocks:
            x2 = [x2] * len(x1)
            out1 += x1
            out2 += x2
        else:
            out1, out2 = x1, x2

        return out1, attn_weight1, pos1, masks1, out2, attn_weight2, pos2, masks2


    def forward(self, img1, img2, do_mask=False, **kwargs):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size
        do_mask: mask only img1.  
        kwarges contains inference argument for attention weight-based matching task: 
          reciprocity=True, suppress_1st_token=False, attn_layers_adopted:list[int]=None, temperature=0.02
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case 
        """
        # encoder
        if self.get_attn_weight:
            feat1, attn_weight1, pos1, masks1, feat2, attn_weight2, pos2, masks2 = self._encode_image(
                img1, img2, do_mask, False, False, kwargs.get("reciprocity", True)
            )
        else:
            feat1, pos1, masks1, feat2, pos2, masks2 = self._encode_image(
                img1, img2, do_mask, False, False, kwargs.get("reciprocity", True)
            )
        # decoder
        if self.num_reg_tokens > 0 and not self.dec_with_reg:
            feat1, pos1, masks1, attn_weight1 = self.exclude_register_tokens(
                feat1, pos1, masks1, attn_weight1 if self.get_attn_weight else None
            )
        
        decfeat1 = self._decode(feat1, pos1, masks1, False)

        if self.num_reg_tokens > 0 and self.dec_with_reg:
            feat1, pos1, masks1, attn_weight1 = self.exclude_register_tokens(
                feat1, pos1, masks1, attn_weight1 if self.get_attn_weight else None
            )
        if self.get_attn_weight and kwargs.get("reciprocity", True):
            feat2, pos2, masks2, attn_weight2 = self.exclude_register_tokens(
                feat2, pos2, masks2, attn_weight2
            )

        # prediction head
        out1 = self.prediction_head(decfeat1)

        # get target
        target1 = patchify(img1, self.patch_size)

        if not self.get_attn_weight:
            return out1, masks1, target1
        else:
            attn_weight, corresp, flow = self.process_attn_weight(attn_weight1, attn_weight2, **kwargs)
            return out1, masks1, target1, attn_weight, corresp, flow