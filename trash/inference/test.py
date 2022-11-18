
import torch
import pickle as pickle
import torch.nn.functional as F
import pandas as pd
import numpy as np
from argparse import Namespace
from model.model_selection import Selection
from data_preprocessing.preprocessing import Preprocessing


class Test():
    def __init__(self, config: Namespace, preprocessing: Preprocessing):
        ## Device
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        ## Setting
        self.config = config
        
        ## Test data
        self.test_data = preprocessing.get_test_data()
        self.test_text = self.test_data["sentence"]
        
        ## Get model and tokenizer
        selection = Selection(config)
        self.model = selection.get_model()
        self.tokenizer = selection.get_tokenizer()
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.to(self.device)
        
        ## Store
        self.test_label_store = []
        self.test_prob_store = []
        
    def test(self):
        for i in range(len(self.test_text)):
            now_text = self.test_text[i]
        
            out = self.tokenizer.encode_plus(
                    now_text,
                    max_length=self.config.mx_token_size,
                    truncation=True,
                    pad_to_max_length=True,
                    add_special_tokens=True,
                )
                
            idz = [out["input_ids"]]
            attentions = [out["attention_mask"]]
            token_types = [out["token_type_ids"]]

            idz = torch.tensor(idz).to(self.device)
            attentions = torch.tensor(attentions).to(self.device)
            token_types = torch.tensor(token_types).to(self.device)
            
            with torch.no_grad():
                pred = self.model(idz, attentions, token_types)
            
            prob = F.softmax(pred, dim=-1).detach().cpu().numpy()
            result = np.argmax(prob, axis=-1)
            
            self.test_label_store.append(result)
            self.test_prob_store.append(prob.tolist()[0])
        
        self.num_to_label()
        self.make_submission_file()
        
        
    def num_to_label(self):
        with open("../code/dict_num_to_label.pkl", "rb") as f:
            dict_num_to_label = pickle.load(f)
        
        decoded_label = []
        for i in range(len(self.test_label_store)):
            idx = self.test_label_store[i].tolist()[0]
            decoded_label.append(dict_num_to_label[idx])
        
        self.test_data["pred_label"] = decoded_label
        self.test_data["probs"] = self.test_prob_store
        
    def make_submission_file(self):
        final_output = self.test_data.loc[:, ["id", "pred_label", "probs"]]
        final_output.to_csv(self.config.result_path, index=False)