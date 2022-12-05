import torch
import pandas as pd
from typing import Tuple
from argparse import Namespace
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
        
        if self.config.model_type == 3: # R-BERT
            out["sub_entity_mask"] = [0] * len(out["input_ids"])
            out["obj_entity_mask"] = [0] * len(out["input_ids"])
            
            s_s = self.data["new_sub_start"][idx]
            tok_s_s = out.char_to_token(self.data["new_sub_start"][idx])
            while tok_s_s is None:
                s_s += 1
                tok_s_s = out.char_to_token(s_s)
            
            s_e = self.data["new_sub_end"][idx]
            tok_s_e = out.char_to_token(self.data["new_sub_end"][idx])
            while tok_s_e is None:
                s_e -= 1
                tok_s_e = out.char_to_token(s_e)
            
            o_s = self.data["new_obj_start"][idx]
            tok_o_s = out.char_to_token(self.data["new_obj_start"][idx])
            while tok_o_s is None:
                o_s += 1
                tok_o_s = out.char_to_token(o_s)
            
            o_e = self.data["new_obj_end"][idx]
            tok_o_e = out.char_to_token(self.data["new_obj_end"][idx])
            while tok_o_e is None:
                o_e -= 1
                tok_o_e = out.char_to_token(o_e)

            for i in range(tok_s_s, tok_s_e + 1):
                if i is None:
                    continue
                out["sub_entity_mask"][i] = 1
            
            for i in range(tok_o_s, tok_o_e + 1):
                if i is None:
                    continue
                out["obj_entity_mask"][i] = 1
            
            out['sub_entity_mask'] = torch.tensor(out['sub_entity_mask'])
            out['obj_entity_mask'] = torch.tensor(out['obj_entity_mask'])

        return out
        
        
    def __len__(self) -> int: 
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

        if self.config.model_type == 3: # R-BERT
            out["sub_entity_mask"] = [0] * len(out["input_ids"][0])
            out["obj_entity_mask"] = [0] * len(out["input_ids"][0])
            
            s_s = self.data["new_sub_start"][idx]
            tok_s_s = out.char_to_token(self.data["new_sub_start"][idx])
            while tok_s_s is None:
                s_s += 1
                tok_s_s = out.char_to_token(s_s)
            
            s_e = self.data["new_sub_end"][idx]
            tok_s_e = out.char_to_token(self.data["new_sub_end"][idx])
            while tok_s_e is None:
                s_e -= 1
                tok_s_e = out.char_to_token(s_e)
            
            o_s = self.data["new_obj_start"][idx]
            tok_o_s = out.char_to_token(self.data["new_obj_start"][idx])
            while tok_o_s is None:
                o_s += 1
                tok_o_s = out.char_to_token(o_s)
            
            o_e = self.data["new_obj_end"][idx]
            tok_o_e = out.char_to_token(self.data["new_obj_end"][idx])
            while tok_o_e is None:
                o_e -= 1
                tok_o_e = out.char_to_token(o_e)

            for i in range(tok_s_s, tok_s_e + 1):
                if i is None:
                    continue
                out["sub_entity_mask"][i] = 1
            
            for i in range(tok_o_s, tok_o_e + 1):
                if i is None:
                    continue
                out["obj_entity_mask"][i] = 1
            
            out['sub_entity_mask'] = torch.tensor(out['sub_entity_mask']).unsqueeze(0)
            out['obj_entity_mask'] = torch.tensor(out['obj_entity_mask']).unsqueeze(0)
        
        return out
        
        
    def __len__(self) -> int:
        return len(self.data)
    