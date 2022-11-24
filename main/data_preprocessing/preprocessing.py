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
from transformers import AutoTokenizer
import collections


class Preprocessing():
    """
    전처리를 담당하는 Class
    """    
    def __init__(self, config: Namespace):
        """
        전처리한 데이터를 처리하는 부분

        Args:
            config (Namespace): Setting Parameters
            tokenizer (tokenizer): tokenzier
        """
        ## Setting
        self.config = config
        
        ## Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.mask_id = self.tokenizer.mask_token_id
        
        ## Load dataset & DataLoader
        self.train_data = pd.read_csv(self.config.train_data_path)
        self.test_data = pd.read_csv(self.config.test_data_path)
        self.val_data = pd.read_csv(self.config.val_data_path)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label2num = None
        
        ## Get Label & Label encoding to number
        '''
        mode : '모델'마다 분류해야 하는 개수가 다르기 때문에 만든 변수.
        'base' : 30개의 라벨로 분류.
        'rescent' : 그룹에 맞는 라벨 수로 분류.
        '''
        
        self.set_label2num()
        print("Label has been mapped to :", self.label2num)
        ## Transformer 모델의 linear output 수를 조절하기 위해 변수 추가.
        self.classes = len(self.label2num)
        
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
        if self.config.input_type == 1:
            self.typed_entity_marker_punct_kr(self.train_data)
            self.typed_entity_marker_punct_kr(self.val_data)
            self.typed_entity_marker_punct_kr(self.test_data)
        ## MLM
        elif self.config.input_type == 2:
            self.concat_and_mask(self.train_data)
            self.concat_and_mask(self.val_data)
            self.concat_and_mask(self.test_data)
        #typed_entity_marker_front
        if self.config.input_type == 3:
            self.typed_entity_marker_punct_front(self.train_data)
            self.typed_entity_marker_punct_front(self.val_data)
            self.typed_entity_marker_punct_front(self.test_data)
        
        
        ## Train & Validation Seperation
        ## TODO: Validation dataset Seperation or other method
        #self.seperate_train_val()
        
        ## Make data loader
        self.make_data_set()
        
    def set_label2num(self):
        modes = {0: "base", 1: "rescent"}
        mode = modes[self.config.train_type]
        if mode == "base":
            with open("./source/dict_label_to_num.pkl", "rb") as f:
                self.label2num = pickle.load(f)    
        elif mode == "rescent":
            labels = list(self.train_data["label"].unique()) + list(self.val_data["label"].unique())
            labels = sorted(list(set(labels)))
            self.label2num = {label: i for i, label in enumerate(labels)}
            # save label dict to 
            with open(self.config.label_dict_dir, "wb") as f:
                pickle.dump(self.label2num, f)
        self.label_to_num(self.train_data)
        self.label_to_num(self.val_data)
    
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
            s = eval(i)
            o = eval(j)
            
            s_word = s["word"]
            s_start = s["start_idx"]
            s_end = s["end_idx"]
            s_type = s["type"]
            
            o_word = o["word"]
            o_start = o["start_idx"]
            o_end = o["end_idx"]
            o_type = o["type"]
            
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
            store.append(obj[i]+" [SEP] "+sub[i]+" [SEP] "+sentence[i])
        data["sentence"] = store
    
    def typed_entity_marker_punct_kr(self, data):
        
        dic = {"PER": "사람", "ORG": "조직", "LOC": "장소", "DAT": "일시", "POH": "명사", "NOH": "숫자"}
    
        store = []
        for i in range(len(data)):
            s = data["sentence"][i]
            sj = data["sub_word"][i]
            s_s = int(data["sub_start"][i])
            s_e = int(data["sub_end"][i])
            s_t = data["sub_type"][i]
            oj = data["obj_word"][i]
            o_s = int(data["obj_start"][i])
            o_e = int(data["obj_end"][i])
            o_t = data["obj_type"][i]
            
            subject_entity = "@ " + "+ " + dic[s_t] + " + " + sj + " @ "
            object_entity = "# " + "^ " + dic[o_t] + " ^ " + oj + " # "
            
            if s_e > o_e:
                s1 = s[:o_s]
                s2 = s[o_e+1:s_s]
                s3 = s[s_e+1:]
                new_s = s1 + object_entity + s2 + subject_entity + s3
            else:
                s1 = s[:s_s]
                s2 = s[s_e+1:o_s]
                s3 = s[o_e+1:]
                new_s = s1 + subject_entity + s2 + object_entity + s3
            store.append(new_s)
        data["sentence"] = store
    
    def typed_entity_marker_punct_front(self, data):
        
        #typed_entity_marker_punct_kr에서 일부 수정되었습니다
        #tag : kr -> eng
        #sentence form :
        #   before: @ + PER + 박수현 @ 은 오늘 # ^ LOC ^ 시청 # 에 들렀다
        #   after : @ + PER + 박수현 @[SEP]# ^ LOC ^ 시청 #[SEP] @ + PER + 박수현 @ 은 오늘 # ^ LOC ^ 시청 # 에 들렀다
        
        store = []
        for i in range(len(data)):
            s = data["sentence"][i]
            sj = data["sub_word"][i]
            s_s = int(data["sub_start"][i])
            s_e = int(data["sub_end"][i])
            s_t = data["sub_type"][i]
            oj = data["obj_word"][i]
            o_s = int(data["obj_start"][i])
            o_e = int(data["obj_end"][i])
            o_t = data["obj_type"][i]
            
            subject_entity = "@ " + "+ " + sj + " + " + sj + " @ "
            object_entity = "# " + "^ " + oj + " ^ " + oj + " # "
            
            if s_e > o_e:
                s1 = s[:o_s]
                s2 = s[o_e+1:s_s]
                s3 = s[s_e+1:]
                new_s = subject_entity + " [SEP] " + object_entity + " [SEP] " + s1 + object_entity + s2 + subject_entity + s3
            else:
                s1 = s[:s_s]
                s2 = s[s_e+1:o_s]
                s3 = s[o_e+1:]
                new_s = object_entity + " [SEP] " + subject_entity + " [SEP] " + s1 + subject_entity + s2 + object_entity + s3
            store.append(new_s)
        data["sentence"] = store

    def concat_and_mask(self, data):
        """
        MLM을 위해 관계 부분을 masking한 문장 만드는 함수
        """
        new_sentences = []
        for i in range(len(data)):
            new_sentences.append(
                f'{data["sentence"][i]} {self.tokenizer.sep_token} {data["sub_word"][i]}와 {data["obj_word"][i]}의 관계는 {self.tokenizer.mask_token}'
            )
        data["sentence"] = new_sentences
    
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
        encoded_label = []
        for label in list(data["label"]):
            encoded_label.append(self.label2num[label])
        
        data["encoded_label"] = encoded_label
    
    def make_data_set(self):
        """
        train loader와 validation loader를 생성하는 함수
        """
        self.train_dataset = DataSet(self.train_data, self.tokenizer, self.config, self.label2num)
        self.val_dataset = DataSet(self.val_data, self.tokenizer, self.config, self.label2num)
        self.test_dataset = DataSetTest(self.test_data, self.tokenizer, self.config, self.label2num)
    
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
    def __init__(self, data: DataFrame, tokenizer, config: Namespace, label2num):
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
    def __init__(self, data: DataFrame, tokenizer, config: Namespace, label2num: dict):
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
    