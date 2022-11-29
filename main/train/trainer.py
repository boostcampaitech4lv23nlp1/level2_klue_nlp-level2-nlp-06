import sys
import datetime
import numpy as np
import pandas as pd
import torch
import warnings
import torch.nn.functional as F
import torch.nn as nn
import sklearn
import wandb
from tqdm import tqdm
from pandas import DataFrame
from torch import Tensor
from typing import List
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoConfig, Trainer, TrainingArguments
from .criterion import FocalLoss
from inference.test import Test

## TODO: 이 모듈을 사용해 loss function을 변경해볼 수 있다.
class CustomTrainer(Trainer):
    def __init__(self, weights, loss_type, **args):
        super(CustomTrainer, self).__init__(**args)
        # for weighted CrossEntropy
        self.weights = weights.to("cuda")
        self.loss_type = loss_type

    """
    Huggingface의 trainer의 loss 부분 Overwrite 한 부분

    Args:
        Trainer (Trainer): Huggingface에서 제공해주는 학습 모듈
    """    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Huggingface의 trainer의 loss 부분을 CrossEntropyLoss로 변경한 부분

        Args:
            model (nn.Module): 훈련시킬 모델
            inputs (tuple): 
            return_outputs (bool, optional): return output이 필요한 경우 pred를 return해주고 그렇지 않으면 loss만 return 해준다

        Returns:
            _type_: _description_
        """
        if self.loss_type in [0,1]:
            loss_fn = nn.CrossEntropyLoss(self.weights)
        elif self.loss_type == 2:
            loss_fn = FocalLoss(0.25,2)

        labels = inputs.get("labels")
        
        pred = model(**inputs)
        
        loss = loss_fn(pred, labels)
           
        ## huggingface의 trainer 내부를 보면 outputs[1:] 이 부분이 있다.
        ## 우선 huggingface trainer의 구조를 파악한 이후 근본적인 문제를 해결할 생각이다.
        dummy = [0] * pred.shape[1]
        dummy = torch.Tensor([dummy]).cuda()
        pred = torch.cat([dummy, pred])
        
        return (loss, pred) if return_outputs else loss
        
        
class MyTrainer():
    """
    Train stpe을 구현한 Class
    """
    def __init__(
            self, 
            model, 
            tokenizer,
            train_dataset: Dataset,
            val_dataset: Dataset,
            val_data: DataFrame,
            config: Namespace,
            weights: torch.Tensor,
        ):
        """`
        Trainer Class의 기본 setting 설정

        Args:
            model (nn.Module): train에 사용할 모델
            tokenizer (tokenizer): tokenizer
            train_loader (Dataset): train dataset
            val_loader (Dataset): validation dataset
            config (NameSpace): Setting
        """
        ## Setting
        self.config = config
        self.weights = weights
        
        ## Device
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        ## Model & Tokenizer
        self.model = model
        self.tokenizer = tokenizer
        
        ## Train & Validation Dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_data = val_data
        
        ## 학습에 필요한 parameter 설정
        ## TODO: 여기에서 여러 가지의 하이퍼파라미터 설정해볼 수 있음
        self.training_args = TrainingArguments(
            output_dir=config.checkpoint_dir,
            save_total_limit=2,
            save_steps=config.eval_step,
            num_train_epochs=config.epoch,
            learning_rate=config.lr,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_steps=config.warmup_step,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=config.eval_step,
            evaluation_strategy='steps',
            eval_steps=config.eval_step,
            load_best_model_at_end=True,
            fp16=True,
        )
        
    def train(self):
        """
        Huggingface 라이브러리를 사용해 학습
        """
        trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            weights=self.weights,
            loss_type=self.config.loss_type
        )
        
        trainer.train()
        self.save()
    
    def save(self): 
        torch.save(self.model.state_dict(), self.config.save_path)
    
    def curriculum(self, k):
        self.model.to(self.device)
        self.model.eval()
        
        self.dataloader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        
        store = []
        for data in tqdm(self.dataloader):
            data = {k: v.squeeze().to(self.device) for k, v in data.items()}
            
            with torch.no_grad():
                pred = self.model(**data)
                prob = F.softmax(pred, dim=-1).detach().cpu()
                
            store.append(prob)
        store = torch.cat(store, dim=0)
        label_list = torch.argmax(store, dim=-1)
        
        return label_list
    
    def curriculum_maker(self, label_lists, fold_data, train_data):
        
        ## Check the answer
        data["0"] = -1
        data["1"] = -1
        data["2"] = -1
        data["3"] = -1
        data["4"] = -1
        data["final"] = 0
        for k in range(5):
            now = str(k)
            unused_data, used_data = fold_data[k]
            ids = list(unused_data["id"])
            for i in range(len(label_lists)):
                data[now].iloc[ids] = label_lists[k]
        
        ## Get Final score
        for i in range(len(data)):
            total = 0
            now = train_data["encoded_label"]
            
            if data["0"].iloc[i] != -1 and data["0"].iloc[i] == now:
                total+=1
            if data["1"].iloc[i] != -1 and data["1"].iloc[i] == now:
                total+=1
            if data["2"].iloc[i] != -1 and data["2"].iloc[i] == now:
                total+=1
            if data["3"].iloc[i] != -1 and data["3"].iloc[i] == now:
                total+=1
            if data["4"].iloc[i] != -1 and data["4"].iloc[i] == now:
                total+=1

            data["final"].iloc[i] = total

        ## Sort the data
        train_data = train_data.sort_values(by=["final"])
        
        ## Make initial data
        train_data = train_data[["id", "sentence", "subject_entity", "object_entity", "label", "source"]]
        
        ## Save Data
        train_data.to_csv("curriculum_data.csv", index=False)
    
    def klue_re_micro_f1(self, preds, labels):
        """KLUE-RE micro f1 (except no_relation)"""
        label_list = list(self.train_dataset.label2num.keys()) # get label_list from train_dataset.
        label_indices = list(range(len(label_list)))
        if "no_relation" in label_list:
            no_relation_label_idx = label_list.index("no_relation")
            label_indices.remove(no_relation_label_idx)
        return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

    def klue_re_auprc(self, probs, labels):
        """KLUE-RE AUPRC (with no_relation)"""
        num_classes = len(self.train_dataset.label2num) # get length of label_list from train-dataset.
        labels = np.eye(num_classes)[labels]

        score = np.zeros((num_classes,))
        for c in range(num_classes):
            targets_c = labels.take([c], axis=1).ravel()
            preds_c = probs.take([c], axis=1).ravel()
            precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
            score[c] = sklearn.metrics.auc(recall, precision)
        return np.average(score) * 100.0
    
    def compute_metrics(self, pred):
        """ validation을 위한 metrics function """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = pred.predictions
        
        ## 저장
        self.save()
        
        # calculate accuracy using sklearn's function
        f1 = self.klue_re_micro_f1(preds, labels)
        auprc = self.klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)
        
        return {
            'micro f1 score': f1,
            'auprc' : auprc,
            'accuracy': acc,
        }