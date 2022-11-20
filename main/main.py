import os
import argparse
import random
import torch
import warnings
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader
from model.model_selection import Selection
from data_preprocessing.preprocessing import Preprocessing
from train.trainer import MyTrainer
from inference.test import Test


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
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--num_hidden_layer", type=int)
    parser.add_argument("--mx_token_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--input_type", type=int)
    parser.add_argument("--model_type", type=int)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_group", type=str)
    
    ## Set seed
    set_seeds(6)
    
    ## Reset the memory
    torch.cuda.empty_cache()
    
    ## Parameters
    config = parser.parse_args()
    config.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    ## Wandb
    #wandb.init(project=config.wandb_project, name=config.wandb_name, notes="binary classification", entity=config.wandb_entity, group=config.wandb_group)
    
    ## Get transformer & tokenizer
    selection = Selection(config)
    model = selection.get_model()
    tokenizer = selection.get_tokenizer()
    
    ## Data preprocessing
    preprocessing = Preprocessing(config, tokenizer)
    train_dataset = preprocessing.get_train_dataset()
    val_dataset = preprocessing.get_val_dataset()
    test_dataset = preprocessing.get_test_dataset()
    
    ## Training
    trainer = MyTrainer(model, tokenizer, train_dataset, val_dataset, config)
    
    print("-----------------Start Training-----------------")
    trainer.train()
    print("-----------------Finish Training-----------------")
    
    ## Testing
    test = Test(config, test_dataset)
    print("&&&&&&&&&&& Start Testing &&&&&&&&&&&")
    test.test()
    print("&&&&&&&&&&& Finish &&&&&&&&&&&")
        