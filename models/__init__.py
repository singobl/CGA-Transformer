#%%
from .unet import UNet
from .transunet import CGA_Transformer
import torch.nn as nn
import torch
import numpy as np

def get_model(residual_num):
    channel = 11
    model = CGA_Transformer(channel, 3, dim=64, residual_num=16, norm='in', dropout=0.0, skip_res=True)
    return model
