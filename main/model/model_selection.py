import torch.nn as nn

from argparse import Namespace
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
from model.models import TransformerModel, TransformerModelUsingMask


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
        self.model = None
        
        ## TODO: 이 부분 좀 더 깔끔하게. 나중에 GPT나 LSTM도 추가되면 그 때 하기.
        transformer_models = [0, 1]
        gpt_models = []
        lstm_models = []
        ## BERT model - Linear.
        if self.config.model_type in transformer_models:
            ## Classification with [ClS] or [MASK]
            ## Model config setting
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(self.config.model_name, config=model_config)

            ## Get final model
            self.model = TransformerModel(transformer, self.config)
            self.model.config = model_config
        ## MLM
        elif self.config.model_type == 2:
            ## Model config setting
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(
                self.config.model_name,
                config=model_config,
                add_pooling_layer=False)

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            ## Get final model
            self.model = TransformerModelUsingMask(transformer, self.tokenizer.mask_token_id, self.config)
            self.model.config = model_config

        ## TODO: 다른 모델을 사용할 경우
        '''
        ## GPT model.
        elif self.config.model_type in gpt_models:
            ...
        ## LSTM model.
        elif self.config.model_type in lstm_models:
            ...
        '''
    
    def add_unk_token(self):
        pass
    
    ## Return model & tokenizer
    def get_model(self): return self.model
    