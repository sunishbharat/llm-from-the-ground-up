
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from transformer_block  import transformer_block
from transformer_components import LayerNormalization, MultiHeadCasualFast, ShortcutConnections,Feedforward


class GPTModel(nn.Module):
    def __init__(self, cfg=config.GPT_CONFIG_124M):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embd_dim"])
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["embd_dim"])
        self.drop    = nn.Dropout(cfg["dropout_rate"])
        
        #Create Transformer blocks
        self.transf_blk = nn.Sequential(
           *[transformer_block(cfg) for _ in range(cfg["num_layers"])]
        )
        
        #Create Layernormalisation layer
        self.layer_norm = LayerNormalization(cfg["embd_dim"])
        
        #Create final output layer
        self.out = nn.Linear(cfg["embd_dim"], cfg["vocab_size"], bias=False)
        
    def forward1(self,x):
        batch_size, seq_len = x.shape
        print(f"{batch_size=},{seq_len=}")

    def forward(self,x):
        batch_size, seq_len = x.shape
        print(f"{batch_size=},{seq_len=}")
        tok_embds = self.tok_emb(x)
        pos_embds = self.pos_emb(torch.arange(seq_len,device=x.device))
        x = tok_embds + pos_embds
        x = self.drop(x)
        x = self.transf_blk(x)
        x = self.layer_norm(x)
        logits = self.out(x)
        return logits
