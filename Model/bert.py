import torch
from transformers import *

class Bert:
    def __init__(self, pretrained_config_name, pretrained_model_name, pretrained_vocab_name):
        bert_config = BertConfig.from_pretrained(pretrained_config_name)
        self.bert_model = BertModel.from_pretrained(pretrained_model_name, config = bert_config)
        self.bert_tok   = BertTokenizer.from_pretrained(pretrained_vocab_name)

    def getModel(self):
        return self.bert_model
    
    def getTokenizer(self):
        return self.bert_tok
