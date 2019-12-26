import Model.bert as bert 
import os
import torch

root_dir = os.getcwd()
bert_config_name = os.path.join(root_dir, 'PretrainedModel/bert-base-uncased-config.json')
bert_model_name = os.path.join(root_dir, 'PretrainedModel/bert-base-uncased-pytorch_model.bin')
bert_vocab_name = os.path.join(root_dir, 'PretrainedModel/bert-base-uncased-vocab.txt')

modelobj = bert.Bert(bert_config_name, bert_model_name, bert_vocab_name)
model = modelobj.getModel()
tok = modelobj.getTokenizer()

text = 'this is the text'
text_idx = torch.LongTensor([tok.encode(text)])
print(text_idx)
out = model(text_idx)[0]
print(out.shape)