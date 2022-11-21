import torch
import pickle as pickle
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pandas import DataFrame
from argparse import Namespace
from model.model_selection import Selection
from data_preprocessing.preprocessing import Preprocessing
from torch.utils.data import Dataset, DataLoader

class Test():
    """
    Test 모듈
    """    
    def __init__(self, config: Namespace, test_dataset: Dataset, test_data: DataFrame):
        ## Device
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        ## Setting
        self.config = config
        
        ## Test data
        self.test_dataset = test_dataset
        self.test_data = test_data
        
        ## Get model and tokenizer
        selection = Selection(config)
        self.model = selection.get_model()
        self.tokenizer = selection.get_tokenizer()
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.to(self.device)
        
        ## Store
        self.test_label_store = []
        self.test_prob_store = []
        
        self.dataloader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        
    def test(self):
        
        for i in range(len(self.test_dataset)):
            out = self.test_dataset[i]
            out = out.to(self.device)

            with torch.no_grad():
                pred = self.model(**out)
            
            prob = F.softmax(pred["logits"], dim=-1).detach().cpu().numpy()
            result = np.argmax(prob, axis=-1)
            
            self.test_label_store.append(result)
            self.test_prob_store.append(prob.tolist()[0])
            
        self.num_to_label()
        self.make_submission_file()
        
    def num_to_label(self):
        """
        숫자로 encoding되어 있는 label을 실제 label로 decoding해주는 함수
        """        
        with open("./source/dict_num_to_label.pkl", "rb") as f:
            dict_num_to_label = pickle.load(f)
        
        decoded_label = []
        for i in range(len(self.test_label_store)):
            idx = self.test_label_store[i].tolist()[0]
            decoded_label.append(dict_num_to_label[idx])
        
        self.test_data["pred_label"] = decoded_label
        self.test_data["probs"] = self.test_prob_store
        
    def make_submission_file(self):
        """
        최종 파일을 저장하는 함수
        """        
        final_output = self.test_data.loc[:, ["id", "pred_label", "probs"]]
        final_output.to_csv(self.config.result_path, index=False)