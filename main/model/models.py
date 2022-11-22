import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, transformer, config):
        super(TransformerModel, self).__init__()
        ## Setting
        self.config = config
        
        ## Transformer model
        self.transformer = transformer
        
        ## Dimension
        self.h_dim = self.transformer.config.hidden_size
        
        ## Layers
        layers = []
        for i in range(self.config.num_hidden_layer): 
            layers.append(nn.Linear(self.h_dim, self.h_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.h_dim, self.config.num_labels))
        self.sequence = nn.Sequential(*layers)
        
    def forward(self, *input, input_ids, token_type_ids, attention_mask, labels=None):
        ## Tranformer output
        x = self.transformer(
            input_ids, #kwargs["input_ids"],
            token_type_ids = token_type_ids, #kwargs["token_type_ids"],
            attention_mask = attention_mask, #kwargs["attention_mask"],
            return_dict = True,
        ).pooler_output
        
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
        layers.append(nn.Linear(self.h_dim, self.transformer.config.num_labels))
        self.sequence = nn.Sequential(*layers)
        
    def forward(self, *input, input_ids, token_type_ids, attention_mask, labels=None):
        ## Tranformer output
        x = self.transformer(
            input_ids, #kwargs["input_ids"],
            token_type_ids = token_type_ids, #kwargs["token_type_ids"],
            attention_mask = attention_mask, #kwargs["attention_mask"],
            return_dict = True,
        ).last_hidden_state

        x_mask = torch.stack([h[i.tolist().index(self.mask_token_id)] for h, i in zip(x, input_ids)])
        
        ## Classification layer
        out = self.sequence(x_mask)
        
        return out
