import torch.nn as nn

from argparse import Namespace
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification, AutoModelForMaskedLM
from model.models import TransformerModel, TransformerModelUsingMask


class Selection():
    """ 
    Select model
    """
    def __init__(self, config: Namespace, mask_id):
        """ 
        Initalization

        Args:
            config (Namespace): 모든 설정 값을 저장하고 있는 것
            mask_id(int): Masked_QA를 위한 tokenizer의 mask_id.
        """
        ## Parameters
        self.config = config
        self.mask_id = mask_id
        
        ## Initialize transformer & tokenizer & model
        transformer = None
        self.model = None
        
        ## BERT model - Linear.
        if self.config.model_type == 0:
            ## Classification with [ClS] or [MASK]
            ## Model config setting
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(self.config.model_name, config=model_config)

            ## Get final model
            self.model = TransformerModel(transformer, self.config)
            self.model.config = model_config
        ## Masked QA
        elif self.config.model_type == 1:
            ## Model config setting
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(
                self.config.model_name,
                config=model_config,
                add_pooling_layer=False)
            
            ## Get final model
            self.model = TransformerModelUsingMask(transformer, self.mask_id, self.config)
            self.model.config = model_config
        ## MLM Pretraining
        elif self.config.model_type == 2:
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            transformer = AutoModelForMaskedLM.from_pretrained(
                self.config.model_name,
                config=model_config
            )

        ## TODO: 다른 모델을 사용할 경우
        '''
        ## GPT model.
        elif self.config.model_type == "GPT":
            ...
        ## LSTM model.
        elif self.config.model_type in lstm_models:
            ...
        '''
    
    def add_unk_token(self):
        pass
    
    ## Return model & tokenizer
    def get_model(self): return self.model
    