import os
import torch
import wandb
import random
import inspect
import warnings
import argparse
import numpy as np
import pandas as pd
from parser import arg_parser
from inference.test import Test
from train.trainer import MyTrainer
from model.model_selection import Selection
from torch.utils.data import Dataset, DataLoader
from data_preprocessing.preprocessing import Preprocessing


def set_seeds(seed=random.randrange(1, 10000)):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    ## Argument Parser
    parser = argparse.ArgumentParser()
    parser = arg_parser(parser)
    
    ## Parameters
    config = parser.parse_args()
    
    ## Set seed
    set_seeds(6)
    
    ## Reset the memory
    torch.cuda.empty_cache()
    
    ## Wandb
    wandb.init(project=config.wandb_project, name=config.wandb_name, notes=config.wandb_note, entity=config.wandb_entity, group=config.wandb_group)
    
    ## Data preprocessing
    preprocessing = Preprocessing(config)
    train_dataset = preprocessing.get_train_dataset()
    val_dataset = preprocessing.get_val_dataset()
    test_dataset = preprocessing.get_test_dataset()
    val_data = preprocessing.get_val_data()
    test_data = preprocessing.get_test_data()
    
    ## Get transformer & tokenizer
    ## TODO: preprocessing config에 넣기
    selection = Selection(config, preprocessing.mask_id)
    model = selection.get_model()
    
    ## Curriculum
    if config.train_type == 2:
        store = []
        for k in range(5):
            selection = Selection(config, preprocessing.mask_id)
            model = selection.get_model()
            train_dataset, val_dataset = preprocessing.get_fold_dataset()[k]
            
            trainer = MyTrainer(
                model=model,
                tokenizer=preprocessing.tokenizer,
                train_dataset=val_dataset, 
                val_dataset=train_dataset,
                val_data=val_data,
                config=config, 
                weights=preprocessing.weights
            )
            print("-----------------Start Training-----------------")
            trainer.train()
            print("-----------------Finish Training-----------------")
            
            label_list = trainer.curriculum(k)
            store.append(label_list)
            
        trainer.curriculum_maker(store, preprocessing.get_fold_data(), preprocessing.get_train_data())
    
    ## Kfold
    elif config.train_type == 3:
        result_path = config.result_path
        for k in range(5):
            selection = Selection(config, preprocessing.mask_id)
            model = selection.get_model()
            train_dataset, val_dataset = preprocessing.get_fold_dataset()[k]
            
            trainer = MyTrainer(
                model=model, 
                tokenizer=preprocessing.tokenizer, 
                train_dataset=train_dataset, 
                val_dataset=val_dataset, 
                val_data=val_data, 
                config=config, 
                weights=preprocessing.weights
            )
            print("-----------------Start Training-----------------")
            trainer.train()
            print("-----------------Finish Training-----------------")
            
            config.result_path = result_path + "-fold" + str(k)
            test = Test(config, test_dataset, test_data)
            print("&&&&&&&&&&& Start Testing &&&&&&&&&&&")
            test.test()
            print("&&&&&&&&&&& Finish &&&&&&&&&&&")

    ## Normal Train
    else:
        ## Training
        trainer = MyTrainer(
            model=model, 
            tokenizer=preprocessing.tokenizer, 
            train_dataset=train_dataset, 
            val_dataset=val_dataset, 
            val_data=val_data, 
            config=config, 
            weights=preprocessing.weights
        )
        
        print("-----------------Start Training-----------------")
        trainer.train()
        print("-----------------Finish Training-----------------")
        
        ## Testing
        test = Test(config, test_dataset, test_data)
        print("&&&&&&&&&&& Start Testing &&&&&&&&&&&")
        test.test()
        print("&&&&&&&&&&& Finish &&&&&&&&&&&")
        