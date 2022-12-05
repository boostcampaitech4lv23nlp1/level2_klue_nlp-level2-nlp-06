import torch.nn as nn

from argparse import Namespace
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification, AutoModelForMaskedLM
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
        self.mask_id = config.mask_id
        
        ## Initialize transformer & tokenizer & model
        transformer = None
        self.model = None
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        
        ## BERT model - Linear.
        if self.config.model_type == 0:
            transformer = AutoModel.from_pretrained(self.config.model_name, config=model_config)

            self.model = TransformerModel(transformer, self.config)
            self.model.config = model_config
        ## Masked QA
        elif self.config.model_type == 1:
            transformer = AutoModel.from_pretrained(self.config.model_name, config=model_config, add_pooling_layer=False)
            
            self.model = TransformerModelUsingMask(transformer, self.mask_id, self.config)
            self.model.config = model_config
        ## MLM Pretraining
        elif self.config.model_type == 2:
            transformer = AutoModelForMaskedLM.from_pretrained(
                self.config.model_name,
                config=model_config
            )
        ## R-BERT
        elif self.config.model_type == 3:
            if self.config.entity_from == "middle":
                model_config.output_hidden_states = True
            model_config.num_labels = 30
            transformer = AutoModel.from_pretrained(self.config.model_name, config=model_config, add_pooling_layer=False)

            self.model = TransformerModelUsingEntity(transformer, config=model_config, entity_from=self.config.entity_from)
            self.model.config = model_config
    
    ## Return model & tokenizer
    def get_model(self): 
        return self.model
    