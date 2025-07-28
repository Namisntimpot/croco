# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------



import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from models.blocks import Block, PatchEmbed
from models.blocks import DecoderBlock_Matching as DecoderBlock
from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from models.masking import RandomMask


class CroCoNet(nn.Module):

    def __init__(self,
                 img_size=224,           # input image size
                 patch_size=16,          # patch_size 
                 mask_ratio=0.9,         # ratios of masked tokens 
                 enc_embed_dim=768,      # encoder feature dimension
                 enc_depth=12,           # encoder depth 
                 enc_num_heads=12,       # encoder number of heads in the transformer block 
                 dec_embed_dim=512,      # decoder feature dimension 
                 dec_depth=8,            # decoder depth 
                 dec_num_heads=16,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='cosine',     # positional embedding (either cosine or RoPE100)
                ):
                
        super(CroCoNet, self).__init__()
                
        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # mask generations
        self._set_mask_generator(self.patch_embed.num_patches, mask_ratio)

        self.pos_embed = pos_embed
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

        # transformer for the encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # masked tokens 
        self._set_mask_token(dec_embed_dim)

        # decoder 
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)
        
        # prediction head 
        self._set_prediction_head(dec_embed_dim, patch_size)
        
        # initializer weights
        self.initialize_weights()           

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)
        
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)
        
    def _set_prediction_head(self, dec_embed_dim, patch_size):
         self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)
        
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None: torch.nn.init.normal_(self.mask_token, std=.02)
        # linears and layer norms
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
            
    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        image has B x 3 x img_size x img_size 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        B,N,C = x.size()
        if do_mask:
            masks = self.mask_generator(x)  # B,N
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B,N,C = x.size()
            masks = torch.zeros((B,N), dtype=bool)
            posvis = pos
        # now apply the transformer encoder and normalization        
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks
 
    def _decoder(self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
                           
        masks1 can be None => assume image1 fully visible  
        attn_maps: ${num_dec_layers} cross attention maps, each with shape of B*num_heads*N*N 
        """
        # encoder to decoder layer 
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # append masked tokens to the sequence
        B,Nenc,C = visf1.size()
        if masks1 is None: # downstreams
            f1_ = visf1
        else: # pretraining 
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B * Nenc, C)
        # add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1_
        out2 = f2 
        attn_maps = []
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2, attn_map = blk(_out, out2, pos1, pos2)
                out.append(_out)
                attn_maps.append(attn_map)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2, attn_map = blk(out, out2, pos1, pos2)
                attn_maps.append(attn_map)
            out = self.dec_norm(out)
        return out, attn_maps

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs

    def forward(self, img1, img2, reciprocity=True, suppress_1st_token=False, attn_layers_adopted:list[int]=None, temperature=0.02, **kwargs):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size  
        reciprocity: take turns using img1 and img2 as reference image for inference, and then calculate the average attention map.  
        suppress_1st_attn: replace the correlation score of the 1st token with the minimal value. The trick is introduced in ZeroCo  
        attn_layers_adopted: attention maps from which decoder layers should be adopted? this argument should be a list of int.    
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case  
        it will additionally output attention maps in the decoder. the shape is B*N*N, the 1st N belongs to the image to be completed, 2nd N the reference image.
        attention_map: B*num_patches*num_patches,  
        corresp & flow: B*2*(img_size//patch_size)*(img_size//patch_size)  
        """
        b, c, h, w = img1.shape
        # encoder of the masked first image
        feat1, pos1, mask1 = self._encode_image(img1, do_mask=self.training and not reciprocity) # mask the image only during pretraining and reciprocity set to False.
        # encoder of the second image 
        feat2, pos2, _ = self._encode_image(img2, do_mask=False)
        # decoder, attn_maps 
        decfeat_img1, attn_maps_img2_as_ref = self._decoder(feat1, pos1, mask1, feat2, pos2) # attn_maps: list of B*num_heads*N*N, dim=-2: tokens of img1.
        attn_maps_img2_as_ref = self._merge_attn_map(attn_maps_img2_as_ref, suppress_1st_token, attn_layers_adopted)
        if reciprocity:
            decfeat_img2, attn_maps_img1_as_ref = self._decoder(feat2, pos2, None, feat1, pos1)
            attn_maps_img1_as_ref = self._merge_attn_map(attn_maps_img1_as_ref, suppress_1st_token, attn_layers_adopted)
            attn_maps = (attn_maps_img2_as_ref + attn_maps_img1_as_ref.transpose(-1, -2)) / 2  # B*n_patches*n_patches
        else:
            attn_maps = attn_maps_img2_as_ref
        
        corresp, flow = self._weighted_sum_attn_map(
            attn_maps.view(b, h//self.patch_embed.patch_size, w//self.patch_embed.patch_size, -1), temperature
        )  # corresp: B,2,h//patch_size,w//patch_size

        # prediction head 
        out = self.prediction_head(decfeat_img1)
        # get target
        target = self.patchify(img1)
        # 
        return out, mask1, target, attn_maps, corresp, flow
    
    def _merge_attn_map(self, attn_maps, suppress_1st_attn=False, attn_layers_adopted:list[int]=None):
        attn_maps = attn_maps if attn_layers_adopted is None else [attn_maps[idx] for idx in attn_layers_adopted]
        attn_maps = torch.stack(attn_maps, dim=1)  # B, n_blk, n_heads, N, N. dim=-2: tokens of img1  
        attn_maps = torch.mean(attn_maps, dim=(1,2)) # B, N, N
        if suppress_1st_attn:
            attn_maps[:,:,0] = attn_maps.min()  # reference to ZeroCo
        return attn_maps
    
    def _weighted_sum_attn_map(self, corr:torch.Tensor, beta=0.02):
        r'''
        SFNet: Learning Object-aware Semantic Flow (Lee et al.)  
        corr: B*(img_size//patch_size)*(img_size//patch_size) * num_patches, the (img_size//patch_size)*(img_size//patch_size) indicates source image tokens.
        beta: temperature.  
        '''
        b,h,w,p = corr.shape
        corr = F.softmax(corr, dim=-1)
        corr = corr.view(-1, h, w, h, w)  # (source hxw) x (target hxw)  

        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=corr.device), torch.arange(w, device=corr.device), indexing='ij')  # hxw
        grid = torch.stack([grid_x, grid_y], dim=0).float()  # grid coordinate, 2xhxw

        corresp = torch.sum(corr.unsqueeze(-3) * grid, dim=(-2,-1))  # b*2*source_h*source_w
        flow = corresp - grid

        return corresp, flow


class MMAE_CroCoNet(CroCoNet):
    def __init__(self, 
                 img_size=224, patch_size=16, mask_ratio=0.9, enc_embed_dim=768, 
                 enc_depth=12, enc_num_heads=12, dec_embed_dim=512, dec_depth=8, dec_num_heads=16, 
                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=0.000001), 
                 norm_im2_in_dec=True, pos_embed='cosine'):
        super().__init__(img_size, patch_size, mask_ratio, enc_embed_dim, enc_depth, 
                         enc_num_heads, dec_embed_dim, dec_depth, dec_num_heads, mlp_ratio, 
                         norm_layer, norm_im2_in_dec, pos_embed)
        

    def _decoder(self, feat1, pos1, masks1, feat2, pos2, masks2, return_all_blocks=False):
        '''
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
                           
        masks1 & masks2 can be None => assume image1 fully visible 
        '''
        visf1 = self.decoder_embed(feat1)
        visf2 = self.decoder_embed(feat2)
        B, Nenc1, C = visf1.size()
        _, Nenc2, _ = visf2.size()
        if masks1 is None:
            f1_ = visf1
        else:
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B*Nenc1, C)
        if masks2 is None:
            f2_ = visf2
        else:
            Ntotal = masks2.size(1)
            f2_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf2.dtype)
            f2_[~masks2] = visf2.view(B*Nenc2, C)
        # add positional embedding.
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2_ = f2_ + self.dec_pos_embed
        # apply Transformer blocks
        out1 = f1_
        out2 = f2_
        attn_maps_img2_as_ref = []
        attn_maps_img1_as_ref = []
        if return_all_blocks:
            _out1, _out2 = [], []
            for blk in self.dec_blocks:
                tmp_out1, _, attn_map_2_ref = blk(out1, out2, pos1, pos2)  # the second return value is equal to the second argum,ent
                _out1.append(tmp_out1)
                attn_maps_img2_as_ref.append(attn_map_2_ref)
                tmp_out2, _, attn_map_1_ref = blk(out2, out1, pos2, pos1)
                _out2.append(tmp_out2)
                attn_maps_img1_as_ref.append(attn_map_1_ref)
                out1 = tmp_out1
                out2 = tmp_out2
            _out1[-1] = self.dec_norm(_out1[-1])
            _out2[-1] = self.dec_norm(_out2[-1])
            return _out1, _out2, attn_maps_img1_as_ref, attn_maps_img2_as_ref
        else:
            for blk in self.dec_blocks:
                tmp_out1, _, attn_map_2_ref = blk(out1, out2, pos1, pos2)
                tmp_out2, _, attn_map_1_ref = blk(out2, out1, pos2, pos1)
                out1 = tmp_out1
                out2 = tmp_out2
                attn_maps_img2_as_ref.append(attn_map_2_ref)
                attn_maps_img1_as_ref.append(attn_map_1_ref)
            out1 = self.dec_norm(out1)
            out2 = self.dec_norm(out2)
            return out1, out2, attn_maps_img2_as_ref, attn_maps_img1_as_ref
        
    
    def forward(self, img1:torch.Tensor, img2:torch.Tensor, reciprocity=True, suppress_1st_token=False, attn_layers_adopted = None, temperature=0.02, **kwargs):
        '''
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size  
        reciprocity must be true and will be neglected  
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case   
        it will additionally output: 
        merged_attn_maps (B*n_patches_of_img1*n_patches_of_img2), corresp & flow (B*2*h//patch_size*w//patch_size)  
        corresp[b,:,i,j] means the coordinates of the correspondence patch of img1's patch(i,i) on img2  
        '''
        b,c,h,w = img1.shape
        feat1, pos1, mask1 = self._encode_image(img1, do_mask=True)
        feat2, pos2, mask2 = self._encode_image(img2, do_mask=True)
        # decoder
        decfeat1, decfeat2, attn_maps_img2_as_ref, attn_maps_img1_as_ref = self._decoder(feat1, pos1, mask1, feat2, pos2, mask2)

        # process attention map.
        attn_maps_img2_as_ref = self._merge_attn_map(attn_maps_img2_as_ref, suppress_1st_token, attn_layers_adopted)
        attn_maps_img1_as_ref = self._merge_attn_map(attn_maps_img1_as_ref, suppress_1st_token, attn_layers_adopted)
        attn_maps = (attn_maps_img2_as_ref + attn_maps_img1_as_ref.transpose(-1, -2)) / 2
        
        corresp, flow = self._weighted_sum_attn_map(
            attn_maps.view(b, h//self.patch_embed.patch_size, w//self.patch_embed.patch_size, -1), temperature
        )  # B, 2, h//patch_size, w//patch_size

        out1 = self.prediction_head(decfeat1)
        out2 = self.prediction_head(decfeat2)
        # get target
        target1 = self.patchify(img1)
        target2 = self.patchify(img2)
        return out1, mask1, target1, out2, mask2, target2, attn_maps, corresp, flow