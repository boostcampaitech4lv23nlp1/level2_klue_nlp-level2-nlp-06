import torch.nn as nn

from argparse import Namespace
from transformers import AutoTokenizer, AutoModel
from model.models import TransformerModel

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
        
        ## Classification with [ClS] or [MASK]
        if self.config.is_transformer:
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            ## Get final model
            self.model = TransformerModel(transformer, config)
            
        ## Classification with Bi-LSTM or Bi-GRU
        else: 
            ## Write the code here
            pass
    
    def add_unk_token(self):
        pass
    
    ## Return model & tokenizer
    def get_model(self): return self.model
    def get_tokenizer(self): return self.tokenizer