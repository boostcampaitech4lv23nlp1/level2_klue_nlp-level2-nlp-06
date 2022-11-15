import random
import pandas as pd
import pickle as pickle
from typing import Tuple
from torch import Tensor
from argparse import Namespace
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Preprocessing():
    """
    전처리를 담당하는 Class
    """    
    def __init__(self, config: Namespace, tokenizer):
        """
        전처리한 데이터를 처리하는 부분

        Args:
            config (Namespace): Setting Parameters
            tokenizer (tokenizer): tokenzier
        """        
        ## Setting
        self.config = config
        
        ## Tokenizer
        self.tokenizer = tokenizer
        
        ## Load dataset & DataLoader
        self.train_data = pd.read_csv(self.config.train_data_path)
        self.test_data = pd.read_csv(self.config.test_data_path)
        self.val_data = None
        self.train_loader = None
        self.val_loader = None
        
        ## Get Label & Label encoding
        self.label_to_num()
        
        self.seperate_train_val()
        
        ## Undersampling
        if self.config.undersampling_flag:
            pass
        
        ## Make data loader
        self.make_data_loader()
        
        
    def seperate_train_val(self):
        """
        train data와 validation data를 분리하는 함수
        """        
        if self.config.val_data_flag == 0:
            self.train_data, self.val_data = train_test_split(self.train_data, test_size=0.06, random_state=random.randrange(1, 10000))
        elif self.config.val_data_flag == 1:
            train_store = []
            val_store = []
            for i in range(30):
                now_data = self.train_data.loc[self.train_data["encoded_label"] == i]
                percent = 20 / len(now_data)
                train, val = train_test_split(now_data, test_size=percent, random_state=random.randrange(1, 10000))
                
                train_store.append(train)
                val_store.append(val)
            
            train_data = train_store[0]
            val_data = val_store[0]
            for i in range(1, 30):
                train_data = pd.concat([train_data, train_store[i]], axis = 0)
                val_data = pd.concat([val_data, val_store[i]], axis = 0)

            self.train_data = train_data.sample(n=len(train_data), replace=False)
            self.val_data = val_data.sample(n=len(val_data), replace=False)
            
    def label_to_num(self):
        """
        data의 label을 숫자로 encoding하는 함수
        """        
        with open("../code/dict_label_to_num.pkl", "rb") as f:
            dict_label_to_num = pickle.load(f)
        
        encoded_label = []
        for i in range(len(self.train_data)):
            encoded_label.append(dict_label_to_num[self.train_data["label"].iloc[i]])
        
        self.train_data["encoded_label"] = encoded_label
    
    def make_data_loader(self):
        """
        train loader와 validation loader를 생성하는 함수
        """                
        train = DataPicker(self.train_data, self.tokenizer, self.config)
        val = DataPicker(self.val_data, self.tokenizer, self.config)
        a = train[0]
        self.train_loader = DataLoader(
            train,
            shuffle = True,
            batch_size = self.config.batch_size,
        )
        self.val_loader = DataLoader(
            val,
            shuffle = True,
            batch_size = self.config.batch_size,
        )
    
    ## ALL Get Method
    def get_train_loader(self): return self.train_loader
    def get_val_loader(self): return self.val_loader
    def get_train_data(self): return self.train_data
    def get_val_data(self): return self.val_data
    def get_test_data(self): return self.test_data

class DataPicker(Dataset):
    """
    데이터를 처리하여 추출하는 Class
    """    
    def __init__(self, data: DataFrame, tokenizer, config: Namespace):
        """
        설정 값 및 tokenizer를 initializing

        Args:
            data (DataFrame): train data, val data, test data 중 하나
            tokenizer (tokenzier): tokenizer
            config (Namespace): Setting Parameters
        """        
        ## Setting
        self.config = config
        
        ## Data & Tokenizer
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        """
        이 Class를 indexing했을 때 return하는 값을 설정하는 함수

        Args:
            idx (int): 데이터의 index

        Returns:
            List[Tensor, Tensor, Tensor, int]: Tensor는 transformer input으로 들어가고, int는 encoded class
        """              
        out = self.tokenizer.encode_plus(
            list(self.data["sentence"])[idx],
            return_tensors="pt",
            max_length=self.config.mx_token_size,
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        
        input_ids = out["input_ids"][0]
        attention_mask = out["attention_mask"][0]
        token_type_ids = out["token_type_ids"][0]
        
        return input_ids, attention_mask, token_type_ids, list(self.data["encoded_label"])[idx]
    
    def __len__(self) -> int:
        """
        len 함수를 사용했을 때 return하는 값을 계산하는 함수

        Returns:
            int: 이 Dataset의 전체 데이터 길이
        """        
        return len(self.data)
        
        