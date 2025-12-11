import torch
import torch.nn as nn
import sys
import os

# --- PATH HACK: Allow importing from parent directory ---
# This adds 'FastDiff Code' to the python path so we can import 'model', 'config', etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --------------------------------------------------------

from config import cfg
# Import existing blocks from the parent model.py to avoid rewriting them
from model import ResidualConvBlock, UnetDown, UnetUp, EmbedFC

class SelfAttention(nn.Module):
    """
    Multi-head Self Attention Module.
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # Flatten: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # Reshape back: [B, H*W, C] -> [B, C, H*W] -> [B, C, H, W]
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class ContextUnetWithAttention(nn.Module):
    def __init__(self, config=cfg):
        super().__init__()
        self.in_channels = config.CHANNELS
        self.n_feat = config.N_FEAT
        self.n_classes = config.N_CLASSES
        self.img_size = config.IMG_SIZE

        self.init_conv = ResidualConvBlock(self.in_channels, self.n_feat, is_res=True)

        self.down1 = UnetDown(self.n_feat, self.n_feat)
        self.down2 = UnetDown(self.n_feat, 2 * self.n_feat)

        # Bottleneck size: 32 -> 16 -> 8
        self.bot_size = self.img_size // 4 
        
        # --- NEW: Self-Attention Module at Bottleneck (8x8) ---
        self.attention = SelfAttention(2 * self.n_feat, self.bot_size)
        # ------------------------------------------------------

        self.to_vec = nn.Identity()

        self.timeembed1 = EmbedFC(1, 2*self.n_feat)
        self.timeembed2 = EmbedFC(1, 1*self.n_feat)
        self.contextembed1 = EmbedFC(self.n_classes, 2*self.n_feat)
        self.contextembed2 = EmbedFC(self.n_classes, 1*self.n_feat)

        self.up0 = nn.Sequential(
            ResidualConvBlock(2 * self.n_feat, 2 * self.n_feat), 
            nn.GroupNorm(8, 2 * self.n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * self.n_feat, self.n_feat)
        self.up2 = UnetUp(2 * self.n_feat, self.n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * self.n_feat, self.n_feat, 3, 1, 1),
            nn.GroupNorm(8, self.n_feat),
            nn.ReLU(),
            nn.Conv2d(self.n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        
        # Apply Attention here
        hiddenvec = self.to_vec(down2)
        hiddenvec = self.attention(hiddenvec) 

        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) 
        c = c * context_mask
        
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2) 
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out