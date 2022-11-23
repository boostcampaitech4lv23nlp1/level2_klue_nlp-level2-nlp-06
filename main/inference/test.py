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
from tqdm import tqdm


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
        selection = Selection(config, self.test_dataset.tokenizer.mask_token_id)
        self.model = selection.get_model()
        self.tokenizer = self.test_dataset.tokenizer
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.to(self.device)
        self.model.eval()
        
        ## Store
        self.test_prob_store = []
        
        self.dataloader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        
    ## TODO: Test도 배치 단위로 해서 빠르게 수행하기.
    def test(self):
        for data in tqdm(self.dataloader):
            data = {k: v.squeeze().to(self.device) for k, v in data.items()}

            with torch.no_grad():
                pred = self.model(**data)
                prob = F.softmax(pred, dim=-1).detach().cpu()

            self.test_prob_store.append(prob)
            
        self.test_prob_store = torch.cat(self.test_prob_store, dim=0)
        self.test_label_store = torch.argmax(self.test_prob_store, dim=-1)

        self.num_to_label()
        self.make_submission_file()
        
    def num_to_label(self):
        """
        숫자로 encoding되어 있는 label을 실제 label로 decoding해주는 함수
        """        
        
        dict_num_to_label = {k: v for v, k in self.test_dataset.label2num.items()}
        
        decoded_label = []
        for pred in self.test_label_store:
            decoded_label.append(dict_num_to_label[pred.item()])
        
        self.test_data["pred_label"] = decoded_label
        self.test_data["probs"] = self.test_prob_store.tolist()
        
    def make_submission_file(self):
        """
        최종 파일을 저장하는 함수
        """
        final_output = self.test_data.loc[:, ["id", "pred_label", "probs"]]
        final_output.to_csv(self.config.result_path, index=False)
        print("submission file created on", self.config.result_path)
        