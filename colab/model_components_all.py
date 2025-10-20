
import torch
import os
import torch
import urllib.request
import re
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
#import config

GPT_CONFIG_124M = {
    "vocab_size"  : 50257,
    "context_len" : 256,
    "embd_dim"    : 768,
    "num_heads"   : 12,
    "num_layers"  : 12,
    "dropout_rate": 0.,
    "qkv_bias"    : False,
}

######################################
# generate_text function
#
######################################
def generate_text(model, idx, num_samples, ctx_len):
  for _ in range(num_samples):
      idx_curr = idx[:,-ctx_len:]
      with torch.no_grad():
          logits= model(idx_curr)
      #print(f"{logits.shape}")
      idx_pred = logits[:,-1,:]
      #Extract the position index of the largest logits
      pred_tok = torch.argmax(idx_pred, dim=-1, keepdim=True)
      
      #print(tokenizer.decode(idx.squeeze(0).tolist()))
      idx= torch.cat((idx,pred_tok),dim=1)
      
  return idx

# GPT tokenizer class
from torch.utils.data import Dataset, DataLoader

class GPTTokenizer(Dataset):
  def __init__(self,input, context_len, stride):
    self.x = []
    self.y = []

    self.tokenizer = tiktoken.get_encoding("gpt2")
    tokens = self.tokenizer.encode(input)

    for i in range(0, len(tokens)- context_len, stride):
      indata = tokens[i:i+context_len]
      target = tokens[i+1:i+context_len+1]
      self.x.append(torch.tensor(indata))
      self.y.append(torch.tensor(target))

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

  def decode(self,tk_ids):
    return self.tokenizer.decode([id.tolist() for id in tk_ids])


##################################################################################
#
# GPT Dataloader Class
##################################################################################
def create_loader(input_txt, batch_size,
                  context_len, stride, shuffle,
                  drop_last=True, num_workers=0):

  dataset = GPTTokenizer(input_txt, context_len, stride)

  print(f"{batch_size=},{context_len=}, {stride=},{shuffle=},  {num_workers=}" )



  dataloader = DataLoader(dataset, batch_size=batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          num_workers=num_workers)
  return dataloader,dataset
  





######################################
# Shortcut connection layer class
#
######################################
class ShortcutConnections(nn.Module):
  def __init__(self, embd_dim):
    super().__init__()
    self.layers = nn.ModuleList([
      nn.Sequential( nn.Linear(embd_dim, embd_dim), nn.GELU()),
      nn.Sequential( nn.Linear(embd_dim, embd_dim), nn.GELU()),
      nn.Sequential( nn.Linear(embd_dim, embd_dim), nn.GELU()),
      nn.Sequential( nn.Linear(embd_dim, embd_dim), nn.GELU()),
      nn.Sequential( nn.Linear(embd_dim, 1), nn.GELU()),
    ])

  def forward(self,x):
    for layer in self.layers:
      out = layer(x)
      x = x + out if x.shape == out.shape else out
    return x

def print_gradients(model,x):
  #forward pass
  output = model(x)
  target = torch.tensor([[0.]])

  #loss function
  loss = nn.MSELoss()
  loss = loss(output,target)

  #Backward pass
  loss.backward()

  for name,params in model.named_parameters():
    print(f"{name=} \n{params.grad.abs().mean().item()=}") if "weight" in name else print("---")

######################################
# Feedforward layer class
#
######################################
class Feedforward(nn.Module):
  def __init__(self, embd_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(embd_dim, 4*embd_dim),
        nn.GELU(),
        nn.Linear(4*embd_dim, embd_dim),
    )

  def forward(self,x):
    return self.layers(x)


######################################
# Multihead Attentions layer class
#
######################################
# Multihead Attention class with parallel heads
class MultiHeadCasualFast(nn.Module):
  def __init__(self, d_in,d_out,context_len,dropout=0.,num_heads=1, bias=False):
    super().__init__()
    assert( d_out % num_heads == 0),f"d_out cannot be equally divided by num_heads"
    self.d_in         = d_in
    self.d_out        = d_out
    self.num_heads    = num_heads
    self.context_len  = context_len
    self.Dropout      = dropout
    self.bias         = bias
    self.head_dim     = d_out // num_heads

    self.Wq = torch.nn.Linear(d_in, d_out, bias=self.bias)
    self.Wk = torch.nn.Linear(d_in, d_out, bias=self.bias)
    self.Wv = torch.nn.Linear(d_in, d_out, bias=self.bias)
    self.dropoutLayer = nn.Dropout(dropout)
    self.register_buffer(
        "mask",
        torch.triu(torch.ones(self.context_len, self.context_len), diagonal=1)
    )

  def forward(self, x):
    b,n_tokens, d_in = x.shape

    # Create attention scores
    queries = self.Wq(x)
    keys    = self.Wq(x)
    values  = self.Wq(x)

    #Divide the QKV into heads
    # Split the d_out -> into equal num of heads (i.e if 4 -> 2,2)
    self.queries  = queries.view(b,n_tokens,self.num_heads, self.head_dim)
    self.keys     = keys.view(b,n_tokens,self.num_heads, self.head_dim)
    self.values   = values.view(b,n_tokens,self.num_heads, self.head_dim)

    # Split back to [b , num_heads, n_tokens, d_out]
    self.queries= self.queries.transpose(1,2)
    self.keys   = self.keys.transpose(1,2)
    self.values = self.values.transpose(1,2)

    #print(f"{self.queries.shape=}")
    #print(f"{self.keys.shape=}")
    #print(f"{self.values.shape=}")

    # Generate attention scores Q.K ==> .shape = [b, n_tokens, num_heads, head_dims ]
    self.attn_scores_qk = self.queries @ self.keys.transpose(-2,-1)

    #Update masking on the attention scores (inplace operation).
    self.attn_scores_qk.masked_fill_(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf)

    # (Attention weights) Normalise using Softmax = q*k/sqrt(dk)
    self.attn_wts_qk_norm = F.softmax(self.attn_scores_qk/ self.keys.shape[-1]**0.5, dim=-1)

    #update Dropouts if any
    self.attn_wts_qk_norm = self.dropoutLayer(self.attn_wts_qk_norm)

    # Create context vector for output = norm_qk * values
    self.context_vec = (self.attn_wts_qk_norm @ self.values).transpose(-2,-1)

    #print(f"{type(b)=}")
    #print(f"{self.context_len=}")
    #print(f"{self.context_vec.shape=}")

    # Concatenate the Z or context vecs.
    self.context_vec = self.context_vec.contiguous().view(b, self.context_len, self.d_out)

    return self.context_vec
    #return self.keys


######################################
# LayerNormalization class
#
######################################
class LayerNormalization(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self,x):
    self.mean   = x.mean(dim=-1, keepdim=True)
    self.var    = x.var(dim=-1, keepdim=True, unbiased=False)
    self.x_norm = (x - self.mean)/torch.sqrt(self.var + self.eps)
    return self.scale * self.x_norm + self.shift


######################################
# GPTModel class
#
######################################
class GPTModel(nn.Module):
    def __init__(self, cfg=GPT_CONFIG_124M):
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
        
    def forward(self,x):
        batch_size, seq_len = x.shape
        #print(f"{batch_size=},{seq_len=}")
        tok_embds = self.tok_emb(x)
        pos_embds = self.pos_emb(torch.arange(seq_len,device=x.device))
        x = tok_embds + pos_embds
        x = self.drop(x)
        x = self.transf_blk(x)
        x = self.layer_norm(x)
        logits = self.out(x)
        return logits


######################################
# Transformer_block class
#
######################################
class transformer_block(nn.Module):
    def __init__(self, cfg=GPT_CONFIG_124M):
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

