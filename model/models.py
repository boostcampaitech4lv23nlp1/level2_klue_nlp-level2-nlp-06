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
        self.classification_dim = 30
        
        ## Layers
        layers = []
        for i in range(self.config.num_hidden_layer): 
            layers.append(nn.Linear(self.h_dim, self.h_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.h_dim, self.classification_dim))
        self.sequence = nn.Sequential(*layers)
        
    def forward(self, input_ids, attention_masks, token_type_ids):
        ## Tranformer output
        x = self.transformer(
            input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_masks,
            return_dict = True,
        ).pooler_output
        
        ## Classification layer
        out = self.sequence(x)
        return out