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
from torch import Tensor
from typing import List
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoConfig, Trainer, TrainingArguments

## TODO: 이 모듈을 사용해 loss function을 변경해볼 수 있다.
class CustomTrainer(Trainer):
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
        loss_fn = nn.CrossEntropyLoss()
        
        labels = inputs.get("labels")
        
        pred = model(**inputs)
        
        loss = loss_fn(pred, labels)
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
        config: Namespace
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
        
        ## Model & Tokenizer
        self.model = model
        self.tokenizer = tokenizer
        
        ## Train & Validation Dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        ## 학습에 필요한 parameter 설정
        ## TODO: 여기에서 여러 가지의 하이퍼파라미터 설정해볼 수 있음 
        self.training_args = TrainingArguments(
            output_dir="./results",
            save_total_limit=5,
            save_steps=500,
            num_train_epochs=config.epoch,
            learning_rate=config.lr,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',            
            logging_steps=100,
            evaluation_strategy='steps',
            eval_steps=500,
            load_best_model_at_end=True,
            fp16=False,
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
        )
        
        trainer.train()
        self.save()

    def save(self): self.model.save_pretrained(self.config.save_path)
        
    def klue_re_micro_f1(self, preds, labels):
        """KLUE-RE micro f1 (except no_relation)"""
        label_list = ['no_relation', 'org:top_members/employees', 'org:members',
        'org:product', 'per:title', 'org:alternate_names',
        'per:employee_of', 'org:place_of_headquarters', 'per:product',
        'org:number_of_employees/members', 'per:children',
        'per:place_of_residence', 'per:alternate_names',
        'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
        'per:spouse', 'org:founded', 'org:political/religious_affiliation',
        'org:member_of', 'per:parents', 'org:dissolved',
        'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
        'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
        'per:religion']
        no_relation_label_idx = label_list.index("no_relation")
        label_indices = list(range(len(label_list)))
        label_indices.remove(no_relation_label_idx)
        return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

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
    
    def compute_metrics(self, pred):
        """ validation을 위한 metrics function """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = pred.predictions
        # binary classification 에서는 accuracy만 계산.
        if self.config.input_type == 1:
            print("labels :", labels.shape, "preds :", preds.shape)
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc}
            
        # calculate accuracy using sklearn's function
        f1 = self.klue_re_micro_f1(preds, labels)
        auprc = self.klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)
        
        return {
            'micro f1 score': f1,
            'auprc' : auprc,
            'accuracy': acc,
        }