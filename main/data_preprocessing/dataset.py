import pandas as pd
from argparse import Namespace
from typing import Tuple

import torch
from torch.utils.data import Dataset


class DataSet(Dataset):
    """
    데이터를 처리하여 추출하는 Class
    """    
    def __init__(self, data: pd.DataFrame, tokenizer, config: Namespace, label2num):
        """
        설정 값 및 tokenizer를 initializing

        Args:
            data (DataFrame): train data, val data, test data 중 하나
            tokenizer (tokenzier): tokenizer
            config (Namespace): Setting Parameters
        """        
        ## Setting
        self.config = config
        self.label2num = label2num
        
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
    def __init__(self, data: pd.DataFrame, tokenizer, config: Namespace, label2num: dict):
        """
        설정 값 및 tokenizer를 initializing

        Args:
            data (DataFrame): train data, val data, test data 중 하나
            tokenizer (tokenzier): tokenizer
            config (Namespace): Setting Parameters
        """        
        ## Setting
        self.config = config
        self.label2num = label2num
        
        ## Data & Tokenizer
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
    