import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch 
import torch.nn as nn
from config.config import config_hp



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OurLSTM(nn.Module): 
  def __init__(self,max_vocab,embedding_size,hidden_size,n_layers):

    super(OurLSTM,self).__init__()

    self.max_vocab=max_vocab
    self.embedding_size=embedding_size
    self.hidden_size=hidden_size
    self.n_layers=n_layers
    
    self.embed=nn.Embedding(self.max_vocab,self.embedding_size,0)
    self.lstm=nn.LSTM(256,self.hidden_size,num_layers=self.n_layers,batch_first=True,bidirectional=True)
    self.liner1=nn.Linear(self.hidden_size*self.n_layers,64)
    self.liner2=nn.Linear(64,32)
    self.output=nn.Linear(32,1) 
    self.drop_out=nn.Dropout(.5)
    self.relu=nn.ReLU()
    self.bn_1=nn.BatchNorm1d(64)
    self.bn_2=nn.BatchNorm1d(32)

  def forward (self,x):
    x=self.embed(x)
    out,_=self.lstm(x)
    x=self.liner1(out[:,-1,:])
    x=self.bn_1(x)
    x=self.relu(x)
    x=self.drop_out(x)
    x=self.liner2(x)
    x=self.bn_2(x)
    x=self.relu(x)
    x=self.drop_out(x)      
    x=self.output(x)

    return x


class BertModel(nn.Module):
  def __init__(self,bert):
    super (BertModel,self).__init__()
    self.bert=bert
    self.l1=nn.Linear(768,256)
    self.l2=nn.Linear(256,64)
    self.output=nn.Linear(64,1)
    self.rlu=nn.ReLU()
    self.drop_o=nn.Dropout(.25)

  def forward(self,input_ids,attention_mask):
    x=self.bert(input_ids, attention_mask, return_dict = False)[0][:,0]
    x=self.l1(x)
    x=self.rlu(x)
    x=self.drop_o(x)
    x=self.l2(x)
    x=self.rlu(x)
    x=self.drop_o(x)
    x=self.output(x)

    return x


if __name__=="__main__": 
    our_model=OurLSTM(config_hp["MAX_VOCAB"],config_hp["EMBEDDING_SIZE"],config_hp["LSTM_HIDDEN_SIZE"],config_hp["LSTM_N_LAYERS"])
    print(our_model(torch.randint(low=0, high=3000, size=(32, 10))).shape)
