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
    batch_size = 4
    epochs = 200

    model = Transformer(src_vocab_size,dst_vocab_size,pad_idx,d_model,num_layes,heads,d_ff,dropout,max_seq_len)
    model.to(device)

    train_datasets = EnglishChineseDataset(tokenizer,"./data/train.txt",max_seq_len)
    test_datasets = EnglishChineseDataset(tokenizer,"./data/test.txt",max_seq_len)

    train_loader = DataLoader(train_datasets,batch_size,shuffle=True)
    test_loader = DataLoader(test_datasets,batch_size,shuffle=False)

    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    loss_fun = nn.CrossEntropyLoss(ignore_index=pad_idx)

    with tqdm.tqdm(total=epochs) as t:
        for epoch in range(epochs):
            model.train()
            for index,(en_in,de_in,de_label) in enumerate(train_loader):
                en_in,de_in,de_label = en_in.to(device),de_in.to(device),de_label.to(device)
                outputs = model(en_in,de_in)
                preds = torch.argmax(outputs,-1)
                label_mask = de_label!=pad_idx

                correct = preds==de_label
                acc = torch.sum(label_mask*correct)/torch.sum(label_mask)
                # batch seq_len,dst_vocab_size
                outputs_ = outputs.reshape(-1,outputs.shape[-1])
                d_label_ = de_label.reshape(-1)
                train_loss = loss_fun(outputs_,d_label_)

                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()

                if index%100==0:
                    print(f"iter:{index}/{len(train_loader)} train loss = {train_loss.item()} acc = {acc}")
            
            torch.save(model.state_dict(),"model.pt")
            print("successfully save model!")
            model.eval()
            with torch.no_grad():
                for index,(en_in,de_in,de_label) in enumerate(test_loader):
                    en_in,de_in,de_label = en_in.to(device),de_in.to(device),de_label.to(device)
                    outputs = model(en_in,de_in)
                    preds = torch.argmax(outputs,-1)
                    label_mask = de_label!=pad_idx

                    correct = preds==de_label
                    acc = torch.sum(label_mask*correct)/torch.sum(label_mask)
                    # batch seq_len,dst_vocab_size
                    outputs_ = outputs.reshape(-1,outputs.shape[-1])
                    d_label_ = de_label.reshape(-1)
                    test_loss = loss_fun(outputs_,d_label_)

                    print(f"iter:{index}/{len(test_loader)} test loss = {test_loss.item()} acc = {acc}")
                    


        
