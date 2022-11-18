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
        
        ## Model we can use
        self.small = ["klue/roberta-small", "beomi/KcELECTRA-small-v2022", "monologg/koelectra-small-v3-discriminator"]
        self.base = ["klue/roberta-base", "beomi/KcELECTRA-base", "monologg/koelectra-base-v3-discriminator", "beomi/kcbert-base", "jinmang2/kpfbert", "klue/bert-base"]
        self.large = ["klue/roberta-large"]
        
        ## Different dimension according to model size
        self.small_first_dim = 256
        self.base_first_dim = 768
        self.large_first_dim = 1024
        
        ## Initialize transformer & tokenizer & model
        transformer = None
        self.tokenizer = None
        self.model = None
        
        ## Classification with [ClS] or [MASK]
        if self.config.is_transformer:
            ## Load transformer & tokenizer
            transformer = AutoModel.from_pretrained(self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            ## Get first input dimension
            first_dim = self.get_transformer_dim()
            
            ## Get final model
            self.model = TransformerModel(transformer, first_dim, config)
            
        ## Classification with Bi-LSTM or Bi-GRU
        else: 
            ## Write the code here
            pass
    
    def add_unk_token(self):
        pass
    
    def get_transformer_dim(self):
        """ 
        Get first hidden dimension of transformer

        Raises:
            Exception: 설정한 모델 이름이 존재하지 않을 경우 발생
        """        
        model_name = self.config.model_name
        if model_name in self.small: return self.small_first_dim
        elif model_name in self.base: return self.base_first_dim
        elif model_name in self.large: return self.large_first_dim
        else: raise Exception("입력한 모델 이름이 존재하지 않습니다. 모델 이름을 업데이트 해주세요.")
    
    ## Return model & tokenizer
    def get_model(self): return self.model
    def get_tokenizer(self): return self.tokenizer