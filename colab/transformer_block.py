
# Transformer Block that includes all the internal components
# - Layernormalization class.
# - MultiHeadAttention
# - Dropout
# - Shortcutconnection
#
#

import torch
import torch.nn as nn
import torch.nn.functional as FF

import config
from transformer_components import LayerNormalization, MultiHeadCasualFast, ShortcutConnections,Feedforward

class transformer_block(nn.Module):
    def __init__(self, cfg=config):
        super().__init__()
        self.layernorm_1 = LayerNormalization(cfg["embd_dim"])
        self.layernorm_2 = LayerNormalization(cfg["embd_dim"])
        self.multihead_attn = MultiHeadCasualFast(
            d_in=cfg["embd_dim"],
            d_out=cfg["embd_dim"],
            context_len=cfg["context_len"],
            dropout=cfg["dropout_rate"],
            num_heads=cfg["num_heads"],
            bias=cfg["qkv_bias"]
        )
        self.shortcut_conn = ShortcutConnections(cfg["embd_dim"])
        self.feedforward = Feedforward(cfg["embd_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        
    def forward(self, x):
        #First layer of tf
        self.input = x
        x = self.layernorm_1(x)
        x = self.multihead_attn(x)
        x = self.dropout(x)
        x = x + self.input 
        
        #Second layer of Tf
        self.input = x 
        x = self.layernorm_2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + self.input
        
        return x


tf = transformer_block(cfg=config.GPT_CONFIG_124M)
tf([1,3,4])