import sys
import datetime
import numpy as np
import pandas as pd
import torch
import warnings
import torch.nn.functional as F
import torch.nn as nn
import sklearn
from torch import Tensor
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

warnings.filterwarnings("ignore")

class Trainer():
    def __init__(self, model, tokenizer, train_loader, val_loader, config):
        ## Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Config
        self.config = config

        ## Model
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        
        ## Data Loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        ## Optimizer (optimizer 셋팅에 따라서 바꿔주는 것도 필요할지도??)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = self.config.lr,
            eps = 1e-8,
            weight_decay = 0.01,
        )
        
        ## Loss Function
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
    
    def train(self):
        step_count = 0
        train_loss_store = []
        for input_ids, attention_mask, token_type_ids, encoded_label in self.train_loader:
            pred = self.prediction(input_ids, attention_mask, token_type_ids)
            encoded_label = encoded_label.type(torch.LongTensor).to(self.device)
            loss = self.step(pred, encoded_label)
            train_loss_store.append(loss)
            step_count += 1
            
            if not step_count % 120:
                print("Num Step : {}".format(step_count))
                val_loss, f1_score, auprc = self.val()
                print("@@@@@@@@@@ val_loss : {} @@@@@@@@@@".format(val_loss))
                print("@@@@@@@@@@ f1 score : {} @@@@@@@@@@".format(f1_score))
                print("@@@@@@@@@@ auprc : {}    @@@@@@@@@@".format(auprc))
    
    def step(self, pred, encoded_label):
        encoded_label = encoded_label.type(torch.LongTensor).to(self.device)
        loss = self.cross_entropy_loss(pred, encoded_label)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def prediction(self, input_ids: List[Tensor], attention_mask: List[Tensor], token_type_ids: List[Tensor]):
        pred = self.model(
            input_ids.to(self.device),
            attention_mask.to(self.device),
            token_type_ids.to(self.device),
            )
        pred = pred.squeeze()
        
        return pred
    
    def val(self):
        total_loss = 0
        batch_count = 0
        
        store_loss = []
        store_pred = []
        store_prob = []
        store_ground_truth = []
        
        for input_ids, attention_mask, token_type_ids, encoded_label in self.val_loader:
            pred = self.prediction(input_ids, attention_mask, token_type_ids)
            encoded_label = encoded_label.type(torch.LongTensor).to(self.device)
            loss = self.cross_entropy_loss(pred, encoded_label)
            
            prob = F.softmax(pred, dim=-1).detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            result = np.argmax(pred, axis=-1)
            
            store_loss.append(loss.detach().cpu())
            store_pred.append(result)
            store_prob.append(prob)
            store_ground_truth.append(encoded_label.detach().cpu().numpy())
            
            batch_count += 1
            total_loss += loss.detach().cpu()
        
        pred_lst = np.concatenate(store_pred).tolist()
        prob_lst = np.concatenate(store_prob)
        real_lst = np.concatenate(store_ground_truth).tolist()
        
        return total_loss/batch_count, self.klue_re_micro_f1(pred_lst, real_lst), self.klue_re_auprc(prob_lst, real_lst)
    
    def klue_re_auprc(self, probs, labels):
        """KLUE-RE AUPRC (with no_relation)"""
        labels = np.eye(30)[labels]

        score = np.zeros((30,))
        for c in range(30):
            targets_c = labels.take([c], axis=1).ravel()
            preds_c = probs.take([c], axis=1).ravel()
            precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
            score[c] = sklearn.metrics.auc(recall, precision)
        return np.average(score) * 100.0
    
    def klue_re_micro_f1(self, preds, labels):
        """KLUE-RE micro f1 (except no_relation)"""
        no_relation_label_idx = 0
        label_indices = list(range(30))
        label_indices.remove(no_relation_label_idx)
        return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

    def save(self):
        torch.save(self.model.state_dict(), self.config.save_path)