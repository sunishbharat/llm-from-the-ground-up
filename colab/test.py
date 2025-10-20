
from GPTModel import GPTModel
import torch
import tiktoken
import config

def decode_text(model, idx, num_samples, ctx_len):
    for _ in range(num_samples):
        idx_curr = idx[:,-ctx_len:]
        with torch.no_grad():
            logits= model(idx_curr)
        #print(f"{logits.shape}")
        idx_pred = logits[:,-1,:]
        pred_tok = torch.argmax(idx_pred, dim=-1, keepdim=True)
        
        print(tokenizer.decode(idx.squeeze(0).tolist()))
        idx= torch.cat((idx,pred_tok),dim=1)
        
    return idx

# Test code
#torch.manual_seed(123)
txt1 = "This is the first test sentence."
txt2 = "Every sentence has an ending word."

tokenizer = tiktoken.get_encoding("gpt2")
    
sample_txt = "Hello I am a good boy from "
enc = tokenizer.encode(sample_txt)
enc_tensor = torch.tensor(enc).unsqueeze(0)
print(f"{enc_tensor.shape=}")

model = GPTModel(cfg=config.GPT_CONFIG_124M)

# To Fix: context len is hardcoded to 7, seems its not working if this is different from cfd.["context_len"]
outt = decode_text(model,enc_tensor,20,7)
lst = outt.squeeze(0).tolist()
#print(f"{lst=}")
#print(tokenizer.decode(lst))
