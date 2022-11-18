import random
import torch
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
        self.val_data = pd.read_csv(self.config.val_data_path)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        ## Get Label & Label encoding to number
        self.label_to_num(self.train_data)
        self.label_to_num(self.val_data)
        
        ## Seperate obj & subj
        self.preprocessing_dataset(self.train_data)
        self.preprocessing_dataset(self.test_data)
        self.preprocessing_dataset(self.val_data)
        
        ## 어떤 input으로 모델을 학습시킬지 결정하는 구간
        ## TODO: input_type을 설정하여
        if self.config.input_type == 0:
            self.simple_concat(self.train_data)
            self.simple_concat(self.test_data)
            self.simple_concat(self.val_data)
        
        ## Train & Validation Seperation
        ## TODO: Validation dataset Seperation or other method
        #self.seperate_train_val()
        
        ## Make data loader
        self.make_data_set()
        
    
    def preprocessing_dataset(self, data: DataFrame):
        """
        initial dataset 내부의 entity를 사용하기 좋게 변형해줍니다.

        Args:
            data (DataFrame): 전처리를 하고 싶은 데이터
        """
        sub_word = []
        sub_start = []
        sub_end = []
        sub_type = []
        
        obj_word = []
        obj_start = []
        obj_end = []
        obj_type = []
        
        for i,j in zip(data["subject_entity"], data["object_entity"]):
            s = i[1:-1].split(":")
            o = j[1:-1].split(":")
            
            s_word = s[1][2:-14]
            s_start = s[2][1:-11]
            s_end = s[3][1:-8]
            s_type = s[4][2:-1]
            
            o_word = o[1][2:-14]
            o_start = o[2][1:-11]
            o_end = o[3][1:-8]
            o_type = o[4][2:-1]
            
            sub_word.append(s_word)
            sub_start.append(s_start)
            sub_end.append(s_end)
            sub_type.append(s_type)
            
            obj_word.append(o_word)
            obj_start.append(o_start)
            obj_end.append(o_end)
            obj_type.append(o_type)

        data["sub_word"] = sub_word
        data["sub_start"] = sub_start
        data["sub_end"] = sub_end
        data["sub_type"] = sub_type
        
        data["obj_word"] = obj_word
        data["obj_start"] = obj_start
        data["obj_end"] = obj_end
        data["obj_type"] = obj_type
    
    def simple_concat(self, data):
        """
        가장 간단하게 object + [SEP] + subject + [SEP] + sentence 를 조합한 방식
        
        Args:
            data (DataFrame): 전처리를 하고 싶은 데이터
        """
        store = []
        obj = list(data["obj_word"])
        sub = list(data["sub_word"])
        sentence = list(data["sentence"])
        for i in range(len(data)):
            store.append(obj[i]+"[SEP]"+sub[i]+"[SEP]"+sentence[i])
        data["sentence"] = store
    
    def seperate_train_val(self):
        """
        train data와 validation data를 간단하게 분리하는 함수
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
            
    def label_to_num(self, data):
        """
        data의 label을 숫자로 encoding하는 함수
        """        
        with open("./source/dict_label_to_num.pkl", "rb") as f:
            dict_label_to_num = pickle.load(f)
        
        encoded_label = []
        for i in range(len(data)):
            encoded_label.append(dict_label_to_num[data["label"].iloc[i]])
        
        data["encoded_label"] = encoded_label
    
    def make_data_set(self):
        """
        train loader와 validation loader를 생성하는 함수
        """                
        self.train_dataset = DataSet(self.train_data, self.tokenizer, self.config)
        self.val_dataset = DataSet(self.val_data, self.tokenizer, self.config)
        self.test_dataset = DataSetTest(self.test_data, self.tokenizer, self.config)
    
    ## ALL Get Method
    def get_train_dataset(self): return self.train_dataset
    def get_val_dataset(self): return self.val_dataset
    def get_test_dataset(self): return self.test_dataset
    def get_train_data(self): return self.train_data
    def get_val_data(self): return self.val_data
    def get_test_data(self): return self.test_data

class DataSet(Dataset):
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
        self.labels = list(data["encoded_label"])

    def __getitem__(self, idx: int):
        """
        이 Class를 indexing했을 때 return하는 값을 설정하는 함수

        Args:
            idx (int): 데이터의 index

        Returns:
            Dict: Tensor는 transformer input으로 들어가고, int는 encoded class
        """              
        out = self.tokenizer.encode_plus(
            list(self.data["sentence"])[idx],
            return_tensors="pt",
            max_length=self.config.mx_token_size,
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        out["input_ids"] = out["input_ids"][0]
        out["token_type_ids"] = out["token_type_ids"][0]
        out["attention_mask"] = out["attention_mask"][0]
        out["labels"] = torch.tensor(self.labels[idx])
        
        return out
        
    def __len__(self) -> int:
        """
        len 함수를 사용했을 때 return하는 값을 계산하는 함수

        Returns:
            int: 이 Dataset의 전체 데이터 길이
        """        
        return len(self.data)
        

class DataSetTest(Dataset):
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
        #self.labels = list(data["encoded_label"])

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
        out["input_ids"] = out["input_ids"]
        out["token_type_ids"] = out["token_type_ids"]
        out["attention_mask"] = out["attention_mask"]
        
        return out
        
    def __len__(self) -> int:
        """
        len 함수를 사용했을 때 return하는 값을 계산하는 함수

        Returns:
            int: 이 Dataset의 전체 데이터 길이
        """        
        return len(self.data)