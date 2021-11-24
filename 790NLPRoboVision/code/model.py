from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

hidden_dim = 128
num_head = 8
num_attention_layers = 6

class MultiviewTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.self_attention_layers = nn.ModuleList([
        	SelfAttention() 
        	for i in range(num_attention_layers)])

        self.cross_attention_layers = nn.ModuleList([
        	CrossAttention() 
        	for i in range(num_attention_layers)])

        self.layer_norm = nn.LayerNorm(hidden_dim)

	def forward(self, x_l, x_r):
        b, c, h, w = feat_left.shape

        pos_enc = None
        for i in range(len(self.self_attentionn_layers)):

        	# self attention for left and right images separately
        	x_l = self.layer_norm(x_l)
        	x_l, self_l_w = self.self_attention_layers(x_l, pos_enc)

        	x_r = self.layer_norm(x_r)
        	x_r, self_r_w = self.self_attention_layers(x_r, pos_enc)

        	# cross attention between each left pixel and all right pixels
        	x_l = self.layer_norm(x_l)
        	x_r = self.layer_norm(x_r)
        	x_l, x_r, cross_w = self.self_attention_layers(x_r, x_l, pos_enc)

        disparity = cross_w


        return disparity


class SelfAttention(nn.MultiheadAttention):
    def __init__(self):
    	super(SelfAttention, self)

    def compute_attention(query, key, value, pos_enc):

    	return x, w

	def forward(self, x, pos_enc, mask):
    	q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        # TODO: compute self attention values

        x_after, self_w = self.compute_attention(query=x, key=x, value=x, pos_enc=pos_enc)
        x = x + x_after
    	return x, self_w


class CrossAttention(nn.MultiheadAttention):
    def __init__(self):
    	super(CrossAttention, self)

    def compute_attention(query, key, value, pos_enc):

    	return x, w

    def forward(self, x_l, x_r, pos_enc, mask):
    	
    	x_l_after, cross_w = self.compute_attention(query=x_r, key=x_l, value=x_l, pos_enc=pos_enc)

    	x_l = x_l + x_l_after

    	return x_l, x_r, cross_w

