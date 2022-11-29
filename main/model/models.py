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
        
        if self.pooling == "CLS":
            # 'pooler_output'이 있는 bert 모델은, 
            if 'pooler_output' in x.keys():
                x = x.pooler_output
            # 'pooler_output'이 없는 bert 이외의 모델은, (Electra)
            else:
                # Get CLS token output from last hidden_state
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

        x_mask = torch.stack([h[i.tolist().index(self.mask_token_id)] for h, i in zip(x, input_ids)])
        
        ## Classification layer
        out = self.sequence(x_mask)
        
        return out


class TransformerModelUsingEntity(nn.Module):
    """
    R-BERT
    """
    def __init__(self, transformer, config, entity_from):
        super(TransformerModelUsingEntity, self).__init__()
        ## Setting
        self.config = config
        self.entity_from = entity_from
        
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
        self.classifier_fc = nn.Linear(self.h_dim * 3, self.config.num_labels)
        
    def forward(self, *input, input_ids, token_type_ids, attention_mask, sub_entity_mask, obj_entity_mask, labels=None):
        ## Tranformer output
        if self.entity_from == 'middle':
            x = self.transformer(
                input_ids, #kwargs["input_ids"],
                token_type_ids = token_type_ids, #kwargs["token_type_ids"],
                attention_mask = attention_mask, #kwargs["attention_mask"],
                return_dict = True,
            ) # (batch size, length of sentences, hidden state dim.)

            h0 = x.last_hidden_state[:, 0, :]
            h1 = torch.bmm(sub_entity_mask.unsqueeze(1).float(), x.hidden_states[7]).squeeze(1) / (sub_entity_mask != 0).sum(dim=1).unsqueeze(1)
            h2 = torch.bmm(obj_entity_mask.unsqueeze(1).float(), x.hidden_states[7]).squeeze(1) / (obj_entity_mask != 0).sum(dim=1).unsqueeze(1)
        elif self.entity_from == 'last':
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
