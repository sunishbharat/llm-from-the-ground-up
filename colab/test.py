
from GPTModel import GPTModel
import torch
import tiktoken
import config

def decode_text(model, idx, num_samples, ctx_len):
    for _ in range(num_samples):
        idx_curr = idx[:,-ctx_len:]
        #print(f"{idx.shape=}")
        #print(f"{idx_curr.shape=}")
        with torch.no_grad():
            logits= model(idx_curr)

        print(f"{logits.shape}")
        idx_pred = logits[:,-1,:]
        pred_tok = torch.argmax(idx_pred, dim=-1, keepdim=True)
        
        idx= torch.cat((idx,pred_tok),dim=1)
        
    return idx

# Test code
#torch.manual_seed(123)
txt1 = "This is the first test sentence."
txt2 = "Every sentence has an ending word."

tokenizer = tiktoken.get_encoding("gpt2")
#batch = []
#batch.append(torch.tensor(tokenizer.encode(txt1)))
#batch.append(torch.tensor(tokenizer.encode(txt2)))
#batch = torch.stack(batch, dim=0)
#
#print(batch.shape)
#model = GPTModel(cfg=config.GPT_CONFIG_124M)
#output= model(batch)
#print(f"{output.shape=}")

#if 0:
#    new_tok = decode_text(model,batch,4,7)
#    print(tiktoken.decode(new_tok))
#    
#else:
#    output_list = output[-1,-1,:]
#    max_out = torch.argmax(output_list, dim=-1, keepdim=True).tolist()
#    #print(f"{max_out.shape=},{max_out=}")
#    #aa = max_out.tolist()
#    predicted_txt = tokenizer.decode(max_out)
#    print(f"{predicted_txt=}")
#    #decode_text(output[-1,-1])
    
    
sample_txt = "Hello I am a good boy from "
enc = tokenizer.encode(sample_txt)
enc_tensor = torch.tensor(enc).unsqueeze(0)
print(f"{enc_tensor.shape=}")

model = GPTModel(cfg=config.GPT_CONFIG_124M)
#output= model(enc_tensor)


# To Fix: context len is hardcoded to 7, seems its not working if this is different from cfd.["context_len"]
outt = decode_text(model,enc_tensor,10,7)
lst = outt.squeeze(0).tolist()
print(f"{lst=}")
print(tokenizer.decode(lst))
