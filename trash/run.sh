#!/bin/bash
python3 main.py \
--model_name=klue/bert-base \
--batch_size=16 --epoch=3 \
--num_hidden_layer=5 --mx_token_size=256 \
--lr=5e-5 \
--is_transformer=True --bi_lstm=$False --bi_gru=$False \
--undersampling_flag=$False --mx_label_size=500 \
--val_data_flag=1 --training_type=0 \
--train_data_path=../dataset/train/train_80.csv --test_data_path=../dataset/valid/valid_20.csv \
--save_path=../saved_model/ex.pt --result_path=../result/submission.csv \

