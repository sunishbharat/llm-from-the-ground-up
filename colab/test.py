import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from GPTModel import GPTModel
import config

if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size"  : 50257,
        "context_len" : 256,
        "embd_dim"    : 768,
        "num_heads"   : 12,
        "num_layers"  : 12,
        "dropout_rate": 0.,
        "qkv_bias"    : False,
    }
    

  ################################################################################ 
  # decode_text
  # 
  #
  ################################################################################ 
    def decode_text(model, idx, num_samples, ctx_len):
        for _ in range(num_samples):
            idx_curr = idx[:,-ctx_len:]
            with torch.no_grad():
                logits= model(idx_curr)
            idx_pred = logits[:,-1,:]
            #Extract the position index of the largest logits
            pred_tok = torch.argmax(idx_pred, dim=-1, keepdim=True)
            
            print(tokenizer.decode(idx.squeeze(0).tolist()))
            idx= torch.cat((idx,pred_tok),dim=1)
            
        return idx

  ################################################################################ 
  # main 
  # 
  #
  ################################################################################ 


    torch.manual_seed(123)
    model = GPTModel(cfg=GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)


    context_len = GPT_CONFIG_124M["context_len"]
    outt = decode_text(model,encoded_tensor,50,context_len)
    lst = outt.squeeze(0).tolist()