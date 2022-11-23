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
        layers.append(nn.Linear(self.h_dim, self.transformer.config.num_labels))
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

class TransformerModelUsingEntity(nn.Module):
    def __init__(self, transformer, config):
        super(TransformerModelUsingEntity, self).__init__()
        ## Setting
        self.config = config
        
        ## Transformer model
        self.transformer = transformer
        
        ## Dimension
        self.h_dim = self.transformer.config.hidden_size

        ## Layers
        self.cls_activation = nn.Tanh()
        self.cls_dropout = nn.Dropout(0.1)
        self.cls_fc = nn.Linear(self.h_dim, self.h_dim)

        self.entity_activation = nn.Tanh()
        self.entity_dropout = nn.Dropout(0.1)
        self.entity_fc = nn.Linear(self.h_dim, self.h_dim)

        self.classifier_dropout = nn.Dropout(0.1)
        self.classifier_fc = nn.Linear(self.h_dim * 3, self.transformer.config.num_labels)
        
    def forward(self, *input, input_ids, token_type_ids, attention_mask, sub_entity_mask, obj_entity_mask, labels=None):
        ## Tranformer output
        x = self.transformer(
            input_ids, #kwargs["input_ids"],
            token_type_ids = token_type_ids, #kwargs["token_type_ids"],
            attention_mask = attention_mask, #kwargs["attention_mask"],
            return_dict = True,
        ).last_hidden_state # (batch size, length of sentences, hidden state dim.)

        h0 = x[:, 0, :]
        h1 = torch.bmm(sub_entity_mask.unsqueeze(1).float(), x).squeeze(1) / (sub_entity_mask != 0).sum(dim=1).unsqueeze(1)
        h2 = torch.bmm(obj_entity_mask.unsqueeze(1).float(), x).squeeze(1) / (obj_entity_mask != 0).sum(dim=1).unsqueeze(1)

        h0 = self.cls_activation(h0)
        h0 = self.cls_dropout(h0)
        h0 = self.cls_fc(h0)

        h1 = self.entity_activation(h1)
        h1 = self.entity_dropout(h1)
        h1 = self.entity_fc(h1)

        h2 = self.entity_activation(h2)
        h2 = self.entity_dropout(h2)
        h2 = self.entity_fc(h2)

        h = torch.cat([h0, h1, h2], dim=-1) # (batch size, hidden state dim. * 3)

        h = self.classifier_dropout(h)
        h = self.classifier_fc(h)
        
        return h
