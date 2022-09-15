from torch.utils.data import DataLoader
import pandas as pd
import json
from src.utils.configs import DATA_CSV,DATA_DIRECTORY,DATA_JSON,RUNS_FOLDER
# PyTorch TensorBoard support
import pandas as pd
from transformers import Trainer
from pytorch import nn

class BaseModel(Trainer):
    def __init__(self,model_name:str,Model):
        self.data_csv = pd.read_csv(DATA_CSV)
        self.data_json = json.load(open(DATA_JSON))
        self.DATA_DIRECTORY = DATA_DIRECTORY
        self.model_name = model_name
        self.Model = Model

        
    def create_train_val_test_split(self,train_frac:float=0.8,val_frac:float=0.1):
        train = self.data_csv.sample(frac=train_frac,random_state=200)
        test_and_val = self.data_csv.drop(train.index)
        val = test_and_val.sample(frac=val_frac,random_state=200)
        test = test_and_val.drop(val.index)
        return (train, val, test)
    
    def create_dataloaders(self,train,val,test,batch_size:int=32) -> tuple:
        train_dataset = DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
        val_dataset = DataLoader(dataset=val,batch_size=batch_size,shuffle=False)
        test_dataset = DataLoader(dataset=test,batch_size=batch_size,shuffle=False)
        loaded = (train_dataset,val_dataset,test_dataset)
        return loaded
    


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = pretrained_encoder(...)
        self.decoder = symetric_decoder(...)
        self.linear1 = nn.Linear(dim_encoder_output, dim_intermediary)
        self.linear2 = nn.Linear(dim_intermediary, dim_embedding)
        self.linear3 = nn.Linear(dim_embedding, dim_embedding)
        self.linear4 = nn.Linear(dim_embedding, dim_intermediary)
        self.linear5 = nn.Linear(dim_intermediary,dim_decoder_input)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        return x

    def decode(self,x):
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.decoder(x)
        return x