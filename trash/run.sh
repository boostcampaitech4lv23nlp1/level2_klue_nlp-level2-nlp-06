#!/bin/bash
python3 main.py \
--model_name=klue/roberta-large \
--batch_size=16 --epoch=20 \
--num_hidden_layer=5 --mx_token_size=256 \
--lr=5e-6 \
--is_transformer=True --bi_lstm=$False --bi_gru=$False \
--undersampling_flag=$False --mx_label_size=500 \
--val_data_flag=1 --training_type=0 \
--train_data_path=../dataset/train/train.csv --test_data_path=../dataset/test/test_data.csv \
--save_path=../dataset/pt_model/model-2.pt --result_path=../dataset/submission/sub-2.csv \

