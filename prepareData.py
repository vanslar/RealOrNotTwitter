import pandas as pd
from twitterText import TextPreprocesser
train_file_name = 'Data/train.csv'
test_file_name = 'Data/test.csv'

train_fd = pd.read_csv(train_file_name)
test_fd = pd.read_csv(test_file_name)

text_pres = TextPreprocesser()
train_fd['text_preprocessed'] = train_fd.apply(lambda x:text_pres.preProcess(x.text), axis = 1)
test_fd['text_preprocessed'] = test_fd.apply(lambda x:text_pres.preProcess(x.text), axis = 1)

train_fd.to_csv('Data/train_preprocessed.csv')
test_fd.to_csv('Data/test_preprocessed.csv')