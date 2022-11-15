import os
import argparse
import random
import torch
import warnings
import numpy as np
import pandas as pd

from model.model_selection import Selection
from data_preprocessing.preprocessing import Preprocessing
from train.trainer import Trainer

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
    parser.add_argument("--is_transformer", type=bool)
    parser.add_argument("--undersampling_flag", type=bool)
    parser.add_argument("--mx_label_size", type=int)
    parser.add_argument("--val_data_flag", type=int)
    parser.add_argument("--bi_lstm", type=bool)
    parser.add_argument("--bi_gru", type=bool)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--save_path", type=str)
    
    ## Set seed
    set_seeds()
    
    ## Reset the memory
    torch.cuda.empty_cache()
    
    ## Parameters
    config = parser.parse_args()
    config.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    ## Get transformer & tokenizer
    selection = Selection(config)
    model = selection.get_model()
    tokenizer = selection.get_tokenizer()
    
    ## Data preprocessing
    preprocessing = Preprocessing(config, tokenizer)
    train_loader = preprocessing.get_train_loader()
    val_loader = preprocessing.get_val_loader()
    print(len(preprocessing.get_val_data()))
    print(len(preprocessing.get_train_data()))
    ## Training
    trainer = Trainer(model, tokenizer, train_loader, val_loader, config)
    
    print("-----------------Start Training-----------------")
    for e in range(config.epoch):
        print("########################### Epoch {} Start ###########################".format(e+1))
        trainer.train()
    print("-----------------Finish Training-----------------")