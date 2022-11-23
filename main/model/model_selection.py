import torch.nn as nn

from argparse import Namespace
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
from model.models import TransformerModel, TransformerModelUsingMask, TransformerModelUsingEntity

class Selection():
    """ 
    Select model
    """    
    def __init__(self, config: Namespace):
        """ 
        Initalization

        Args:
            config (Namespace): 모든 설정 값을 저장하고 있는 것
        """
        ## Parameters
        self.config = config
        
        ## Initialize transformer & tokenizer & model
        transformer = None
        self.tokenizer = None
        self.model = None
        
        ## Classification with [CLS] or [MASK]
        if self.config.model_type == 0:
            ## Model config setting
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            model_config.num_labels = 30
            
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(self.config.model_name, config=model_config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            ## Get final model
            #self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, config=model_config)
            self.model = TransformerModel(transformer, config)
            self.model.config = model_config
            
        ## Model classify sentence to "no_relation" or "related"
        elif self.config.model_type == 1:
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            model_config.num_labels = 2
            
            #transformer = AutoModel.from_pretrained(self.config.model_name, config=model_config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, config=model_config)
            #self.model = TransformerModel(transformer, config)
            self.model.config = model_config

        ## MLM
        elif self.config.model_type == 2:
            ## Model config setting
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            model_config.num_labels = 30
            
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(
                self.config.model_name,
                config=model_config,
                add_pooling_layer=False)

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            ## Get final model
            self.model = TransformerModelUsingMask(transformer, self.tokenizer.mask_token_id, config)
            self.model.config = model_config

        ## R-BERT
        elif self.config.model_type == 3:
            ## Model config setting
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            model_config.num_labels = 30
            
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(
                self.config.model_name,
                config=model_config,
                add_pooling_layer=False)

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            ## Get final model
            self.model = TransformerModelUsingEntity(transformer, config)
            self.model.config = model_config
            
        ## TODO: 다른 모델을 사용할 경우
    
    def add_unk_token(self):
        pass
    
    ## Return model & tokenizer
    def get_model(self): return self.model
    def get_tokenizer(self): return self.tokenizer