import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np 
from PIL import Image
from einops.einops import rearrange
from .reg import SemLA_Reg
from .fusion import SemLA_Fusion

class SemLA(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = SemLA_Reg()
        self.fusion = SemLA_Fusion()

    def forward(self, img_vi, img_ir, matchmode):
        # Select 'scene' mode when no semantic objects exist in the image
        if matchmode=='semantic':
            thr = 0.5
        elif matchmode=='scene':
            thr = 0

        """
        feat_reg_vi_final, feat_reg_ir_final [1,256,60,80] N C H W
        feat_sa_vi, feat_sa_ir [1,1,60,80] N C H W
        """
        feat_reg_vi_final, feat_reg_ir_final, feat_sa_vi, feat_sa_ir = self.backbone(
            torch.cat((img_vi, img_ir), dim=0))
        #pdb.set_trace()


        """
        sa_vi, sa_ir [4800] 60x80
        feat_reg_vi,feat_reg_ir [1,4800,256] N HxW C
        """
        sa_vi, sa_ir = feat_sa_vi.reshape(-1), feat_sa_ir.reshape(-1)
        sa_vi, sa_ir = torch.where(sa_vi > thr)[0], torch.where(sa_ir > thr)[0]

        feat_reg_vi = rearrange(feat_reg_vi_final, 'n c h w -> n (h w) c')
        feat_reg_ir = rearrange(feat_reg_ir_final, 'n c h w -> n (h w) c')
        #pdb.set_trace()

        """
        feat_reg_vi, feat_reg_ir [1,4800,256] N HxW C
        """
        feat_reg_vi, feat_reg_ir = feat_reg_vi[:, sa_vi], feat_reg_ir[:, sa_ir]
        feat_reg_vi, feat_reg_ir = map(lambda feat: feat / feat.shape[-1] ** .5,
                                       [feat_reg_vi, feat_reg_ir])
        #pdb.set_trace()


        """
        conf [1,4800,4800]
        mask [1,4800,4800]
        """
        conf = torch.einsum("nlc,nsc->nls", feat_reg_vi,
                            feat_reg_ir) / 0.1
        mask = conf > 0.
        mask = mask \
               * (conf == conf.max(dim=2, keepdim=True)[0]) \
               * (conf == conf.max(dim=1, keepdim=True)[0])
        pdb.set_trace()


        """
        mask_v [1,4800] bool 
        """
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        i_ids = sa_vi[i_ids]
        j_ids = sa_ir[j_ids]

        mkpts0 = torch.stack(
            [i_ids % feat_sa_vi.shape[3], i_ids // feat_sa_vi.shape[3]],
            dim=1) * 8
        mkpts1 = torch.stack(
            [j_ids % feat_sa_vi.shape[3], j_ids // feat_sa_vi.shape[3]],
            dim=1) * 8
        pdb.set_trace()
        sa_ir= F.interpolate(feat_sa_ir, scale_factor=8, mode='bilinear', align_corners=True)
        sa_vi= F.interpolate(feat_sa_vi, scale_factor=8, mode='bilinear', align_corners=True)
        pdb.set_trace()
        return mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir, sa_vi
