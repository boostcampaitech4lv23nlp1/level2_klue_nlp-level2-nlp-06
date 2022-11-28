import torch
import torch.nn as nn


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class TransformerModel(nn.Module):
    def __init__(self, transformer, config):
        super(TransformerModel, self).__init__()
        ## Setting
        self.config = config
        self.pooling = self.config.pooling
        
        ## Transformer model
        self.transformer = transformer
        
        ## Dimension
        self.h_dim = self.transformer.config.hidden_size
        
        ## Layers
        layers = []
        for i in range(self.config.num_hidden_layer):
            layers.append(nn.Linear(self.h_dim, self.h_dim))
            layers.append(nn.ReLU())
            
        if self.config.rnn_type == "lstm":
            self.lstm = nn.LSTM(self.h_dim, self.h_dim, num_layers=2, bias=True, batch_first=True, dropout=0.1, bidirectional=True)
            layers.append(nn.Linear(self.h_dim*2, self.config.num_labels))
        elif self.config.rnn_type == "gru":
            self.gru = nn.GRU(self.h_dim, self.h_dim, num_layers=2, bias=True, batch_first=True, dropout=0.1, bidirectional=True)
            layers.append(nn.Linear(self.h_dim*2, self.config.num_labels))
        else:
            layers.append(nn.Linear(self.h_dim, self.config.num_labels))
        self.sequence = nn.Sequential(*layers)
        
        # BERT가 아닌 Electra, GPT 등의 모델인 경우, Pooling 레이어를 추가.
        if "bert" not in self.config.model_name and self.pooling == "CLS":
            self.pooler_linear = nn.Linear(self.h_dim, self.h_dim)
            self.dropout = nn.Dropout(0.1)
        
    def forward(self, *input, input_ids, token_type_ids, attention_mask, labels=None):
        ## Tranformer output
        x = self.transformer(
            input_ids, #kwargs["input_ids"],
            token_type_ids = token_type_ids, #kwargs["token_type_ids"],
            attention_mask = attention_mask, #kwargs["attention_mask"],
            return_dict = True,
        )
        
        if self.config.rnn_type == "lstm":
            x, (h_n, c_n) = self.lstm(x.last_hidden_state)
            x = x[:, -1, :]
        elif self.config.rnn_type == "gru":
            x, h_n = self.gru(x.last_hidden_state)
            x = x[:, -1, :]
        else:
            if self.pooling == "CLS":
                if 'pooler_output' in x.keys():
                    x = x.pooler_output
                else:
                    x = x.last_hidden_state[:, 0, :]
                    x = self.pooler_linear(x)
                    x = self.dropout(x)
            elif self.pooling == "MEAN":
                x = mean_pooling(x, attention_mask)
        
        ## Classification layer
        out = self.sequence(x)
        
        return out

class TransformerModelUsingMask(nn.Module):
    def __init__(self, transformer, mask_token_id, config):
        super(TransformerModelUsingMask, self).__init__()
        ## Setting
        self.config = config
        self.mask_token_id = mask_token_id
        
        ## Transformer model
        self.transformer = transformer
        
        ## Dimension
        self.h_dim = self.transformer.config.hidden_size
        
        ## Layers
        layers = []
        for i in range(self.config.num_hidden_layer): 
            layers.append(nn.Linear(self.h_dim, self.h_dim))
            layers.append(nn.ReLU())
            
        if self.config.rnn_type == "lstm":
            self.lstm = nn.LSTM(self.h_dim, self.h_dim, num_layers=2, bias=True, batch_first=True, dropout=0.1, bidirectional=True)
        elif self.config.rnn_type == "gru":
            self.gru = nn.GRU(self.h_dim, self.h_dim, num_layers=2, bias=True, batch_first=True, dropout=0.1, bidirectional=True)
        else:
            layers.append(nn.Linear(self.h_dim, self.config.num_labels))
        self.sequence = nn.Sequential(*layers)
        
    def forward(self, *input, input_ids, token_type_ids, attention_mask, labels=None):
        ## Tranformer output
        x = self.transformer(
            input_ids, #kwargs["input_ids"],
            token_type_ids = token_type_ids, #kwargs["token_type_ids"],
            attention_mask = attention_mask, #kwargs["attention_mask"],
            return_dict = True,
        ).last_hidden_state
        
        if self.config.rnn_type == "lstm":
            x, (h_n, c_n) = self.lstm(x)
        elif self.config.rnn_type == "gru":
            x, h_n = self.gru(x.last_hidden_state)

        x_mask = torch.stack([h[i.tolist().index(self.mask_token_id)] for h, i in zip(x, input_ids)])
        
        ## Classification layer
        out = self.sequence(x_mask)
        
        return out
