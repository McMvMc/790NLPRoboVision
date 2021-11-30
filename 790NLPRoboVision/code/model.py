import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from models.feat_ex import FeatEx

use_relative_pe = True
hidden_dim = 128
HIDDEN_DIM = 128
num_freq = hidden_dim//2
num_head = 8
num_attention_layers = 6


class MultiviewTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.use_relative_pe = use_relative_pe
        self.use_ffn_feat = False
        self.pe = None
        self.pe_table = None
        self.feat_ex = FeatEx()

        self.self_attention_layers = nn.ModuleList([
            SelfAttention()
            for i in range(num_attention_layers)])

        self.cross_attention_layers = nn.ModuleList([
            CrossAttention()
            for i in range(num_attention_layers)])

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_l, x_r):
        b, c, h, w = x_l.shape
        device = x_l.device

        # get visual features
        x_l = self.feat_ex(x_l)
        x_r = self.feat_ex(x_r)

        # get positional encodings
        if self.use_relative_pe:
            with torch.no_grad():
                u_tgt = torch.arange(w)[None].to(device)
                u_src = torch.arange(w)[:,None].to(device)
                rel_pos_table = u_tgt - u_src  # actual relative pixel distance, [src_u, tgt_u] is tgt_u-src_u
                rel_pos_enc_index = rel_pos_table + (w-1)  # their code index in the positional encoding table
                n_pos = rel_pos_enc_index.max() + 1
                rel_pos_s2l = torch.arange(n_pos).to(device) - (w-1)

                # get encodings
                freqs = torch.arange(num_freq * 2).to(device)  # will be [1,1,2,2,3,3...] after //2
                dim = 10000**(2*(freqs//2)/hidden_dim)
                pe = rel_pos_s2l[:,None] / dim[None]
                self.pe = torch.stack((pe[:, 0::2].sin(), pe[:, 1::2].cos()), dim=2)
                self.pe_table = torch.index_select(pe, 0, rel_pos_enc_index.view(-1)).view(w, w, -1)  # 2W-1xC -> WW'xC -> WxW'xC

        x_l = x_l.permute(3, 2, 0, 1).flatten(1, 2)
        x_r = x_r.permute(3, 2, 0, 1).flatten(1, 2)

        for i in range(len(self.self_attention_layers)):

            # self attention for left and right images separately
            x_l = self.layer_norm(x_l)

            def call_attn(module):
                def attn(*inputs):
                    return module(*inputs)
                return attn
            x_l, self_l_attn_score = checkpoint(call_attn(self.self_attention_layers[i]),
                                          x_l, self.pe_table, None)

            x_r = self.layer_norm(x_r)
            x_r, self_r_attn_score = checkpoint(call_attn(self.self_attention_layers[i]),
                                          x_r, self.pe_table, None)

            # cross attention between each left pixel and all right pixels
            x_l = self.layer_norm(x_l)
            x_r = self.layer_norm(x_r)

            x_l, x_r, cross_attn_score = checkpoint(call_attn(self.cross_attention_layers[i]),
                                              x_l, x_r, self.pe_table, None)

        u_idx = torch.arange(w).to(device)
        corresp = (F.softmax(cross_attn_score, dim=-1) * u_idx[None, None].expand(h*b, w, -1)).sum(dim=-1)

        disp = (u_idx[None, None].expand(b, h, -1) - corresp.view(b, h, -1)).view(b, h, w)

        return disp


def compute_attention(attn_module, query, key, value, pos_enc_table, is_cross_attn):
    w, h, c = query.size()
    head_dim = c // num_head

    if not is_cross_attn:
        q, k, v = F.linear(query, attn_module.in_proj_weight, attn_module.in_proj_bias).chunk(3, dim=-1)
    else:
        # cross-attention projects right query and left key/value
        # first 1/3 used for query
        q = F.linear(query, attn_module.in_proj_weight[:c, :], attn_module.in_proj_bias[:c])

        # later 2/3 used for key/values
        k, v = F.linear(key, attn_module.in_proj_weight[c:, :], attn_module.in_proj_bias[c:]).chunk(2, dim=-1)

    q = q.view(w, h, num_head, head_dim)
    k = k.view(w, h, num_head, head_dim)
    v = v.view(w, h, num_head, head_dim)

    attn_feat = torch.einsum('whnc,vhnc->hnwv', q, k)  # w=v, h x num_head x w x w

    # include positional encoding
    if pos_enc_table is not None:
        # compute projection for relative encoding
        q_rpe, k_rpe = F.linear(pos_enc_table, attn_module.in_proj_weight[:2*c, :],
                                attn_module.in_proj_bias[:2*c]).chunk(2, dim=-1)  # w x w x c
        q_rpe = q_rpe.view(w, w, num_head, head_dim)  # w x w x num_head x c
        k_rpe = k_rpe.view(w, w, num_head, head_dim)

        attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_rpe)
        attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_rpe)

        # include positional feature
        attn_feat = attn_feat + attn_feat_pos + attn_pos_feat

    assert list(attn_feat.size()) == [h, num_head, w, w]
    attn_feat = attn_feat / (head_dim ** 0.5)
    attn_score = attn_feat.sum(dim=1)
    attn = F.softmax(attn_feat, dim=-1)

    v_out = torch.bmm(attn.flatten(0,1), v.flatten(1,2).permute(1,0,2))

    assert list(v_out.size()) == [h*num_head, w, head_dim]
    v_out = v_out.view(h, attn_module.num_heads, w, head_dim).permute(2, 0, 1, 3).flatten(2, 3)
    v_out = F.linear(v_out, attn_module.out_proj.weight, attn_module.out_proj.bias)

    # attn_avg = attn.sum(dim=1) / attn_module.num_heads
    return v_out, attn_score


class SelfAttention(nn.MultiheadAttention):
    def __init__(self):
        super(SelfAttention, self).__init__(hidden_dim, num_head, bias=True)

    def forward(self, x, pos_enc, mask):

        x_after, attn_score = compute_attention(self, query=x, key=x, value=x, pos_enc_table=pos_enc, is_cross_attn=False)
        x = x + x_after
        return x, attn_score


class CrossAttention(nn.MultiheadAttention):
    def __init__(self):
        super(CrossAttention, self).__init__(hidden_dim, num_head, bias=True)

    def forward(self, x_l, x_r, pos_enc, mask):

        x_l_after, attn_score = compute_attention(self, query=x_r, key=x_l, value=x_l, pos_enc_table=pos_enc, is_cross_attn=True)

        x_l = x_l + x_l_after

        return x_l, x_r, attn_score

