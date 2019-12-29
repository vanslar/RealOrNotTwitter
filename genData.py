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

import pandas as pd
train_file_name = 'Data/train_preprocessed.csv'
test_file_name = 'Data/test_preprocessed.csv'

train_fd = pd.read_csv(train_file_name)
test_fd = pd.read_csv(test_file_name)

def datasetEmbeding(dataset, embeding_model, max_word_count):
    embeding = []
    with torch.no_grad():
        for x in dataset:
            text_idx = torch.LongTensor([tok.encode(x)])
            out = embeding_model(text_idx)[0][0]
            seq_len, feature_len = out.shape
            if seq_len > max_word_count:
                out = out[0:max_word_count, :]
            elif seq_len < max_word_count:
                pad = torch.Tensor([[0]*feature_len] * (max_word_count - seq_len))
                out = torch.cat([out, pad], 0)
            embeding.append(out.numpy())
        return torch.Tensor(embeding)


#train_embeding = datasetEmbeding(train_fd['text_preprocessed'], model, 100)
#test_embeding = datasetEmbeding(test_fd['text_preprocessed'], model, 100)

def genTrainEmbedingBatch(batch_count, epoch_count):
    dataset = train_fd
    ds_len = len(dataset)
    cur_epoch = 0
    cur_batch = 0
    while cur_epoch < epoch_count:
        data = []
        target = []
        if cur_batch + batch_count < ds_len:
            data =  datasetEmbeding(dataset['text_preprocessed'][cur_batch:cur_batch+batch_count], model, 50)
            target = torch.Tensor(dataset['target'][cur_batch:cur_batch+batch_count].values)

            cur_batch += batch_count 
        else:
            data = list(dataset['text_preprocessed'][cur_batch:])
            target = list(dataset['target'][cur_batch:])
            res_count = batch_count - (ds_len - cur_batch)
            data.extend(dataset['text_preprocessed'][:res_count])
            target.extend(dataset['target'][:res_count])

            data =  datasetEmbeding(data, model, 50)
            target = torch.Tensor(target)

            cur_batch = res_count
            cur_epoch += 1
        yield data, target.unsqueeze(1)


def genTestEmbedingBatch(batch_count, epoch_count):
    dataset = test_fd
    ds_len = len(dataset)
    cur_epoch = 0
    cur_batch = 0
    while cur_epoch < epoch_count:
        data = []
        target = []
        if cur_batch + batch_count < ds_len:
            data =  datasetEmbeding(dataset['text_preprocessed'][cur_batch:cur_batch+batch_count], model, 50)

            cur_batch += batch_count 
        else:
            data = dataset['text_preprocessed'][cur_batch:]
            res_count = batch_count - (ds_len - cur_batch)
            data.extend(dataset['text_preprocessed'][:res_count])

            data =  datasetEmbeding(data, model, 50)

            cur_batch = res_count
            cur_epoch += 1
        yield data


#g = genEmbedingBatch(train_fd, 10, 1)
#data, target = next(g)
#print(data.shape, target.shape)