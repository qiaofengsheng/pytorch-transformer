from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
import torch
import random

def split_data(data_path):
    datas = open(data_path,'r',encoding="utf-8").readlines()
    random.shuffle(datas)
    train_data = datas[:int(len(datas)*0.95)]
    test_data =  datas[int(len(datas)*0.95):]
    open("./data/train.txt",'w',encoding="utf-8").writelines(train_data)
    open("./data/test.txt",'w',encoding="utf-8").writelines(test_data)


def count_max_seq_len(data_path):
    datas = open(data_path,'r',encoding="utf-8").readlines()
    max_len = 0
    for data in datas:
        en,zh = data.strip().split("\t")[:2]
        max_len = max(max_len,len(tokenizer(en)["input_ids"]),len(tokenizer(zh)["input_ids"]))
    print(max_len)


class EnglishChineseDataset(Dataset):
    def __init__(self,tokenizer,data_path,max_seq_len=64):
        super().__init__()
        self.tokenizer = tokenizer
        self.datas = open(data_path,'r',encoding="utf-8").readlines()
        self.max_seq_len = max_seq_len
        self.data_cache={}
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        if index in self.data_cache:
            return self.data_cache[index]
        src,dst = self.datas[index].strip().split("\t")[:2]
        src = self.tokenizer(src,padding="max_length",max_length = self.max_seq_len,truncation=True,return_tensors="pt")["input_ids"]
        dst_in = "<s>"+dst
        dst_in = self.tokenizer(dst_in,padding="max_length",max_length = self.max_seq_len,truncation=True,return_tensors="pt")["input_ids"]
        dst_label = self.tokenizer(dst,padding="max_length",max_length = self.max_seq_len,truncation=True,return_tensors="pt")["input_ids"]
        self.data_cache[index]=(torch.LongTensor(src)[0],torch.LongTensor(dst_in)[0],torch.LongTensor(dst_label)[0])
        return torch.LongTensor(src)[0],torch.LongTensor(dst_in)[0],torch.LongTensor(dst_label)[0]

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    tokenizer.add_special_tokens({"bos_token":"<s>"})
    print(tokenizer.bos_token,tokenizer.bos_token_id)
    dataset= EnglishChineseDataset(tokenizer,"./data/train.txt",40)
    print(dataset[0])
    # str_ = "你好，我是一只小小鸟！"
    # out = tokenizer(str_)
    # print(out)
    # count_max_seq_len("./data/cmn.txt")