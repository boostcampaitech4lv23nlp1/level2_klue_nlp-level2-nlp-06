#!/bin/bash
python3 main.py \
--model_name=klue/bert-base \
--batch_size=16 --epoch=10 \
--num_hidden_layer=5 --mx_token_size=256 \
--lr=5e-6 \
--model_type=0 --input_type=0 \
--train_data_path=../dataset/train/train.csv \
--val_data_path=../dataset/train/train.csv \
--test_data_path=../dataset/test/test_data.csv \
--save_path=../dataset/pt_model/model-2.pt --result_path=../dataset/submission/sub-2.csv \
--checkpoint_dir=./results
--wandb_project=koohack --wandb_entity=happy06 --wandb_name=temp --wandb_note=sample\

