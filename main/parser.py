def arg_parser(parser):
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--num_hidden_layer", type=int)
    parser.add_argument("--mx_token_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--input_type", type=int)
    parser.add_argument("--model_type", type=int)
    parser.add_argument("--train_type", type=int)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_group", type=str)
    parser.add_argument("--wandb_note", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--warmup_step", type=int)
    parser.add_argument("--eval_step", type=int)
    parser.add_argument("--label_dict_dir", type=str)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--add_rnn", type=str)
    parser.add_argument("--loss_type", type=int)
    parser.add_argument("--entity_from", type=str)
    return parser