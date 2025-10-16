
import torch
import torch.nn as nn
import torch.nn.functional as F



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

    print(f"{self.queries.shape=}")
    print(f"{self.keys.shape=}")
    print(f"{self.values.shape=}")

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

    print(f"{type(b)=}")
    print(f"{self.context_len=}")
    print(f"{self.context_vec.shape=}")

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

