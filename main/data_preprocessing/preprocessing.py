import pandas as pd
import pickle as pickle
from argparse import Namespace
from transformers import AutoTokenizer
from .dataset import DataSet, DataSetTest
from sklearn.model_selection import StratifiedKFold
from .utils import get_weights_prob, preprocessing_dataset, label_to_num


class Preprocessing():
    """
    전처리를 담당하는 Class
    """    
    def __init__(self, config: Namespace):
        """
        전처리한 데이터를 처리하는 부분

        Args:
            config (Namespace): Setting Parameters
        """
        ## Setting
        self.config = config
        
        ## Tokenizer
        if config.input_type in [4,5,6]: # special token 추가한 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("./data_preprocessing/newtokenizer")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.config.mask_id = self.tokenizer.mask_token_id # QA를 위한 mask id 저장
        
        ## Load dataset & DataLoader
        self.train_data = pd.read_csv(self.config.train_data_path)
        self.test_data = pd.read_csv(self.config.test_data_path)
        self.val_data = pd.read_csv(self.config.val_data_path)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label2num = None
        
        ## Kfold & Curriculum data
        self.folded_data = []
        self.folded_dataset = []
        
        ## Get Label & Label encoding to number
        self.set_label2num()
        self.config.num_labels = len(self.label2num) # Transformer 모델의 linear output 수를 조절하기 위해 변수 추가.
        print("Label has been mapped to :", self.label2num)
        
        ## Get Class Distribution Weights for CrossEntropy loss.
        self.weights = get_weights_prob(self.label2num, self.train_data, self.config.loss_type)
        
        ## Seperate obj & subj
        preprocessing_dataset(self.train_data)
        preprocessing_dataset(self.test_data)
        preprocessing_dataset(self.val_data)
        
        ## 어떤 input으로 모델을 학습시킬지 결정하는 구간
        if self.config.input_type == 0:
            self.simple_concat(self.train_data)
            self.simple_concat(self.test_data)
            self.simple_concat(self.val_data)
        elif self.config.model_type == 3:
            self.create_new_entity_pos(self.train_data)
            self.create_new_entity_pos(self.val_data)
            self.create_new_entity_pos(self.test_data)
        elif self.config.input_type in [1,3,4,5,6]:
            self.entity_marker(self.train_data, config.input_type)
            self.entity_marker(self.val_data, config.input_type)
            self.entity_marker(self.test_data, config.input_type)
            
        ## MLM
        if self.config.model_type == 1:
            self.concat_and_mask(self.train_data)
            self.concat_and_mask(self.val_data)
            self.concat_and_mask(self.test_data)
        
        ## KFold & Curriculum
        if self.config.train_type in [2, 3]:
            self.set_fold()
        
        ## Make data loader
        self.make_data_set()
    
    
    ## TODO: 김준휘
    def set_label2num(self):
        """
        label to num dictionary 정의
        """        
        modes = {0: "base", 1: "rescent", 2: "base", 3: "base"}
        mode = modes[self.config.train_type]
        if mode == "base":
            with open("./source/dict_label_to_num.pkl", "rb") as f:
                self.label2num = pickle.load(f)
        elif mode == "rescent":
            labels = list(self.train_data["label"].unique()) + list(self.val_data["label"].unique())
            labels = sorted(list(set(labels)))
            self.label2num = {label: i for i, label in enumerate(labels)}
            # save label dict to 
            if self.config.label_dict_dir != None:
                with open(self.config.label_dict_dir, "wb") as f:
                    pickle.dump(self.label2num, f)
        self.train_data = label_to_num(self.train_data, self.label2num)
        self.val_data = label_to_num(self.val_data, self.label2num)
        
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
    
    def entity_marker(self, data, input_type):
        """
        input type에 따른 sentence 변경

        Args:
            data (DataFrame): 변경할 dataset
            input_type (int): 'sentence' 유형
        """        
        
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
            
            if input_type == 1:
                subject_entity = "@ " + "+ " + dic[s_t] + " + " + sj + " @ "
                object_entity = "# " + "^ " + dic[o_t] + " ^ " + oj + " # "
            elif input_type == 3:
                subject_entity = "@ " + "+ " + s_t + " + " + sj + " @ "
                object_entity = "# " + "^ " + o_t + " ^ " + oj + " # "
            elif input_type == 4:
                subject_entity = "[SUBJ-"+s_t+"]"
                object_entity = "[OBJ-"+o_t+"]"
            elif input_type == 5:
                subject_entity = " [E1] " + sj + " [/E1] "
                object_entity = " [E2] " + oj + " [/E2] "
            elif input_type == 6:
                subject_entity = " [S:"+s_t+"] " + sj + " [/S:"+s_t+"] "
                object_entity = " [O:"+o_t+"] " + oj + " [/O:"+o_t+"] "

            if s_e > o_e:
                s1 = s[:o_s]
                s2 = s[o_e+1:s_s]
                s3 = s[s_e+1:]
                if input_type != 3:
                    new_s = s1 + object_entity + s2 + subject_entity + s3
                else:
                    new_s = subject_entity + " [SEP] " + object_entity + " [SEP] " + s1 + object_entity + s2 + subject_entity + s3
            else:
                s1 = s[:s_s]
                s2 = s[s_e+1:o_s]
                s3 = s[o_e+1:]
                if input_type != 3:
                    new_s = s1 + subject_entity + s2 + object_entity + s3
                else:
                    new_s = subject_entity + " [SEP] " + object_entity + " [SEP] " + s1 + subject_entity + s2 + object_entity + s3
            store.append(new_s)
        data["sentence"] = store
        

    def concat_and_mask(self, data):
        """
        MLM을 위해 관계 부분을 masking한 문장 만드는 함수

        Args:
            data (DataFrame): train_data, val_data, test_data 중 하나
        """        
        new_sentences = []
        for i in range(len(data)):
            new_sentences.append(
                f'{data["sentence"][i]} {self.tokenizer.sep_token} {data["sub_word"][i]}와 {data["obj_word"][i]}의 관계는 {self.tokenizer.mask_token}'
            )
        data["sentence"] = new_sentences
        

    def create_new_entity_pos(self, data):
        """
        R-Bert input type

        Args:
            data (DataFrame): train_data, val_data, test_data 중 하나
        """        
        ## TODO: entity_marker에서 new position 자체를 생성해 넘겨줄 수 있도록 수정하기
        ## TODO: 설유민
        dic = {"PER": "사람", "ORG": "조직", "LOC": "장소", "DAT": "일시", "POH": "명사", "NOH": "숫자"}

        new_sub_start = []
        new_sub_end = []
        new_obj_start = []
        new_obj_end = []
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
                new_s_s = s_s + 7 + len(dic[s_t]) + 12
                new_s_e = new_s_s + len(sj) - 1
                new_o_s = o_s + 7 + len(dic[o_t])
                new_o_e = new_o_s + len(oj) - 1
            else:
                s1 = s[:s_s]
                s2 = s[s_e+1:o_s]
                s3 = s[o_e+1:]
                new_s = s1 + subject_entity + s2 + object_entity + s3
                new_s_s = s_s + 7 + len(dic[s_t])
                new_s_e = new_s_s + len(sj) - 1
                new_o_s = o_s + 7 + len(dic[o_t]) + 12
                new_o_e = new_o_s + len(oj) - 1
            
            new_sub_start.append(new_s_s)
            new_sub_end.append(new_s_e)
            new_obj_start.append(new_o_s)
            new_obj_end.append(new_o_e)
            store.append(new_s)
        
        data["sentence"] = store
        data["new_sub_start"] = new_sub_start
        data["new_sub_end"] = new_sub_end
        data["new_obj_start"] = new_obj_start
        data["new_obj_end"] = new_obj_end


    def make_data_set(self):
        """
        train loader와 validation loader를 생성하는 함수
        """
        self.train_dataset = DataSet(self.train_data, self.tokenizer, self.config, self.label2num)
        self.val_dataset = DataSet(self.val_data, self.tokenizer, self.config, self.label2num)
        self.test_dataset = DataSetTest(self.test_data, self.tokenizer, self.config, self.label2num)
    
    
    def set_fold(self):
        """
        KFold를 위해 train data를 sub train data와 sub valiadation data로 변경하는 코드
        """        
        skf = StratifiedKFold(n_splits=5)
        for train_index, val_index in skf.split(self.train_data, self.train_data["label"]):
            # train_data, val_data 저장
            train_data, val_data = self.train_data.iloc[train_index], self.train_data.iloc[val_index]
            self.folded_data.append([train_data, val_data])
            
            # trian_dataset, val_dataset 저장
            train_dataset = DataSet(train_data, self.tokenizer, self.config, self.label2num)
            val_dataset = DataSet(val_data, self.tokenizer, self.config, self.label2num)
            self.folded_dataset.append([train_dataset, val_dataset])
        
        
    ## ALL Get Method
    def get_train_dataset(self): return self.train_dataset
    def get_val_dataset(self): return self.val_dataset
    def get_test_dataset(self): return self.test_dataset
    def get_train_data(self): return self.train_data
    def get_val_data(self): return self.val_data
    def get_test_data(self): return self.test_data
    def get_fold_data(self): return self.folded_data
    def get_fold_dataset(self): return self.folded_dataset
