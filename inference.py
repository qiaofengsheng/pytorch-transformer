import torch
from torch import nn,optim
import torch.utils
from model import Transformer
from dataset import *
import tqdm,os

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    tokenizer.add_special_tokens({"bos_token":"<s>"})

    src_vocab_size,dst_vocab_size = tokenizer.vocab_size+len(tokenizer.special_tokens_map),tokenizer.vocab_size+len(tokenizer.special_tokens_map)
    pad_idx=tokenizer.pad_token_id
    print(pad_idx)
    d_model=512
    num_layes=6
    heads=8
    d_ff=1024
    dropout = 0.1
    max_seq_len = 40
    batch_size = 1
    epochs = 200

    model = Transformer(src_vocab_size,dst_vocab_size,pad_idx,d_model,num_layes,heads,d_ff,dropout,max_seq_len)
    model.to(device)

    if os.path.exists("./model.pt"):
        model.load_state_dict(torch.load("./model.pt"))
    
    input_ = "I am Tom."
    
    input_in = tokenizer(input_,padding="max_length",max_length = max_seq_len,truncation=True,return_tensors="pt")["input_ids"]
    print(input_in.shape)
    input_in = input_in.to(device)

    de_in = torch.ones(batch_size,max_seq_len,dtype=torch.long).to(device) * pad_idx

    de_in[:,0] = tokenizer.bos_token_id

    model.eval()
    with torch.no_grad():
        for i in range(1,de_in.shape[1]):
            pred_ = model(input_in,de_in)
            for j in range(batch_size):
                de_in[j,i] = torch.argmax(pred_[j,i-1])
    
    out = []
    for i in de_in[0]:
        if i==tokenizer.eos_token_id:
            break
        out.append(tokenizer.decode(i))
    print(out)




        
