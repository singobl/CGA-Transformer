""" Full assembly of the parts to form the complete network """
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .unet.unet_parts import *
from .layernorm2d import LayerNormConv2d

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math 

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from abc import abstractmethod

from scipy.io import savemat
import os

def silu(x):
    return x * F.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight



class Mlp(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout = 0.):
        super(Attention, self).__init__()
        self.heads = heads
        context_dim = context_dim or query_dim
        hidden_dim = max(query_dim, context_dim)
        # self.dim_head = int(hidden_dim / self.heads)
        self.dim_head = dim_head
        self.all_head_dim = self.heads * self.dim_head

        self.query = nn.Linear(query_dim, self.all_head_dim)
        self.key = nn.Linear(context_dim, self.all_head_dim)
        self.value = nn.Linear(context_dim, self.all_head_dim)
        self.out = nn.Linear(self.all_head_dim, query_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.dim_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, context=None):
        if context is None:
            context = query
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.dim_head)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Block(nn.Module):
    def __init__(self, hidden_size, droppath=0., dropout=0.0):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        self.attn = Attention(hidden_size, dropout=dropout)
        self.drop_path =  nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.attn(self.attention_norm(x))) + x
        x = self.drop_path(self.ffn(self.ffn_norm(x))) + x
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8, save_dir='output'):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.save_dir = save_dir  # Directory to save .mat files

    def forward(self, x, y):
        initial = x + y          
        save_tensor_as_mat(initial, os.path.join(self.save_dir, 'initial.mat'))
        cattn = self.ca(initial) # W_c
        sattn = self.sa(initial) # W_s
        pattn1 = sattn + cattn   
        # Save 'pattn1'
        save_tensor_as_mat(pattn1, os.path.join(self.save_dir, 'Wcoa.mat'))
        
        cs = self.pa(initial, pattn1)  
        # Save 'wcoa'
        save_tensor_as_mat(cs, os.path.join(self.save_dir, 'cs.mat'))
        
        CGAout = self.sigmoid(cs)  
        # Save 'pattn2'
        save_tensor_as_mat(CGAout, os.path.join(self.save_dir, 'CGAout.mat'))
        

        result = initial + CGAout * x + (1 - CGAout) * y
        result = self.conv(result)   
        return result
class CGA_Transformer(nn.Module):
    def __init__(self, n_channels, n_classes, dim=64, residual_num=16, bilinear=True, norm='bn', dropout=0.0, skip_res=True, save_dir='output'):
        super(CGA_Transformer, self).__init__()
        
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip_res = skip_res
        self.fusion   = CGAFusion(11, save_dir=save_dir)
        # Borrow from: https://github.com/ChenyangLEI/sfp-wild
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(dim, dim * 2, norm=norm)
        self.down2 = Down(dim * 2, dim * 4, norm=norm)
        self.down3 = Down(dim * 4, dim * 8, norm=norm)
        self.down4 = Down(dim * 8, dim * 16 // factor, norm=norm)
        
        self.inc1 = DoubleConv(n_channels, dim)
        self.down1_1 = Down(dim, dim * 2, norm=norm)
        self.down2_1 = Down(dim * 2, dim * 4, norm=norm)
        self.down3_1 = Down(dim * 4, dim * 8, norm=norm)
        self.down4_1 = Down(dim * 8, dim * 16 // factor, norm=norm)

        
        self.inc2 = DoubleConv(n_channels, dim)
        self.down1_2 = Down(dim, dim * 2, norm=norm)
        self.down2_2 = Down(dim * 2, dim * 4, norm=norm)
        self.down3_2 = Down(dim * 4, dim * 8, norm=norm)
        self.down4_2 = Down(dim * 8, dim * 16 // factor, norm=norm)

        self.fusion_conv = DoubleConv(dim * 32 // factor, dim * 16 // factor)

        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4 = Up(dim * 2, dim, bilinear)
        self.outc = OutConv(dim, n_classes)
        self.resblock_layers = nn.ModuleList([])
        
        self.save_dir = save_dir  # Directory to save .mat files
        # Missing positinal encoding
        for i in range(residual_num):
            self.resblock_layers.append(Block(dim * 8, dropout=dropout))
    def forward(self, images,est,spec, Iun, cos1, cos2, DoP, image_coordinate):
        x1_input = torch.cat((images, Iun, cos1, cos2, DoP,image_coordinate), dim=1)
        x2_input = torch.cat((est, Iun, spec), dim=1)
        flusion = self.fusion(x1_input, x2_input)  
        b, c, h, w = flusion.size()
        x1 = self.inc(flusion)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b, c, h, w = x5.size()
        x5 = torch.reshape(x5, [b, c, h*w]).permute(0,2,1)
        for resblock in self.resblock_layers:
            residual = resblock(x5)
            if self.skip_res:
                x5 = residual
            else:
                x5 = x5 + residual
        x5 = torch.reshape(x5.permute(0,2,1), [b, c, h, w])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        