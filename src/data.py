import os
import sys  

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ...existing code...

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader ,Dataset
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from src.utils import pad_sentence ,prepare_data,tokenizer
from config.config import config_hp
from transformers import AutoModel ,AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df =pd.read_json(r"D:\Sarcasm_Detection\news-headlines-dataset-for-sarcasm-detection\Sarcasm_Headlines_Dataset.json",lines=True)

df.drop_duplicates(inplace=True)

x_train,x_test,y_train,y_test=train_test_split(np.array(df["headline"]),np.array(df["is_sarcastic"]),test_size=0.2,shuffle=True,random_state=42)


df_tokens=df["headline"].map(prepare_data)


our_vocab=build_vocab_from_iterator(df_tokens,specials=config_hp["SPECIAL_TOKENS"],max_tokens=config_hp["MAX_VOCAB"])
our_vocab.set_default_index(our_vocab[config_hp["SPECIAL_TOKENS"][1]])

class OurDataSet(Dataset):
  def __init__(self,x,y,vocab,max_len):
    self.x=x
    self.y=y
    self.vocab=vocab
    self.max_len=max_len

  def __len__(self):
    return len(self.x)
  def __getitem__(self, index):

    tokens=prepare_data (self.x[index])
    padded=pad_sentence(tokens,self.max_len,config_hp["SPECIAL_TOKENS"][0])
    input_seq=torch.tensor(self.vocab.lookup_indices(padded), dtype=torch.long)
    target=torch.tensor(self.y[index] ,dtype=torch.float)

    return input_seq ,target


our_train_data_set=OurDataSet(x=x_train,y=y_train,vocab=our_vocab,max_len=config_hp["MAX_LEN"])
our_test_data_set=OurDataSet(x=x_test,y=y_test,vocab=our_vocab,max_len=config_hp["MAX_LEN"])

def get_datasets_and_loaders_for_lstm ():
   our_train_data_loader=DataLoader(our_train_data_set,batch_size=config_hp["BATCH_SIZE"],shuffle=True)
   our_test_data_loader=DataLoader(our_test_data_set,batch_size=config_hp["BATCH_SIZE"],shuffle=False)
   return our_train_data_loader,our_test_data_loader,our_train_data_set,our_test_data_set



class BertData(Dataset):
  def __init__(self,x,y):
    self.x=[bert_tokenizer(s,max_length=100,truncation=True,padding="max_length",return_tensors="pt").to(device) for s in x]
    self.y=torch.tensor(y,dtype=torch.float32).to(device)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index) :
    return self.x[index],self.y[index]
  
our_train_data_set_bert=BertData(x=x_train,y=y_train)
our_test_data_set_bert=BertData(x=x_test,y=y_test)

def get_datasets_and_loaders_for_bert ():
   our_train_data_loader_bert=DataLoader(our_train_data_set_bert,batch_size=config_hp["BATCH_SIZE"],shuffle=True)
   our_test_data_loader_bert=DataLoader(our_test_data_set_bert,batch_size=config_hp["BATCH_SIZE"],shuffle=False)
   return our_train_data_loader_bert,our_test_data_loader_bert,our_train_data_set_bert,our_test_data_set_bert




# if __name__ == "__main__":
#     for x,y in our_train_data_loader:
#         print(x.shape)
#         print(y.shape)
#         break

#     for x,y in our_test_data_loader:
#         print(x.shape)
#         print(y.shape)
#         break

