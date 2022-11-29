import pandas as pd
import pickle as pickle
from argparse import Namespace
from transformers import AutoTokenizer

from .utils import get_weights_prob, preprocessing_dataset, label_to_num
from .dataset import DataSet, DataSetTest


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
        if config.input_type in [4,5,6]:
            self.tokenizer = AutoTokenizer.from_pretrained("./data_preprocessing/newtokenizer")
        else:
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
        ## Get Class Distribution Weights for CrossEntropy loss.
        self.weights = get_weights_prob(self.label2num, self.train_data, self.config.loss_type)
        
        ## Seperate obj & subj
        self.preprocessing_dataset = preprocessing_dataset
        self.preprocessing_dataset(self.train_data)
        self.preprocessing_dataset(self.test_data)
        self.preprocessing_dataset(self.val_data)
        
        ## 어떤 input으로 모델을 학습시킬지 결정하는 구간
        ## TODO: input_type을 설정하여
        if self.config.input_type == 0:
            self.simple_concat(self.train_data)
            self.simple_concat(self.test_data)
            self.simple_concat(self.val_data)
        if self.config.input_type in [1,3,4,5,6]:
            self.entity_marker(self.train_data, config.input_type)
            self.entity_marker(self.val_data, config.input_type)
            self.entity_marker(self.test_data, config.input_type)
        ## MLM
        if self.config.model_type == 1:
            self.concat_and_mask(self.train_data)
            self.concat_and_mask(self.val_data)
            self.concat_and_mask(self.test_data)

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
        '''
        input type = 1  typed_entity_marker_punct_kr        :   @ + 사람 + 박수현 @ 은 오늘 # ^ 장소 ^ 시청 # 에 들렀다
        input type = 3  typed_entity_marker_punct_kr_front  :   @ + PER + 박수현 @[SEP]# ^ LOC ^ 시청 #[SEP] @ + PER + 박수현 @ 은 오늘 # ^ LOC ^ 시청 # 에 들렀다
        input type = 4  entity_mask                         :   [SUBJ-PER] 은 오늘 [OBJ-LOC] 에 들렀다
        input type = 5  entity_marker                       :   [E1] 박수현 [/E1]은 오늘 [E2] 시청 [/E2] 에 들렀다
        input type = 6  typed_entity_marker                 :   [S:PER] 박수현 [/S:PER] 은 오늘 [O:LOC] 시청 [/O:LOC] 에 들렀다. 
        '''
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
        """
        new_sentences = []
        for i in range(len(data)):
            new_sentences.append(
                f'{data["sentence"][i]} {self.tokenizer.sep_token} {data["sub_word"][i]}와 {data["obj_word"][i]}의 관계는 {self.tokenizer.mask_token}'
            )
        data["sentence"] = new_sentences
        
            
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
