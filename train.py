from genData import *
from modelUtil import *
from Model.lstm import lstm

g_train = genTrainEmbedingBatch(128, 1000)
model = lstm(768, 128, 1)

train(model, g_train, 'model_check.pth', print_per_loops = 10, save_per_loops = 100)