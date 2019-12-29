import torch
import torch.nn as nn
import os
import numpy as np


def train(model, g, model_checkpoint_name, print_per_loops, save_per_loops):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model = loadModel(model,  model_checkpoint_name)

    model.train()
    print('begin training...')
    critien = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    for ind, inputs in  enumerate(g):
        data, target = inputs
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        out = model(data)
        loss = critien(out, target)
        loss.backward()
        opt.step()
        
        if ind % save_per_loops == 99:
            saveModel(model, model_checkpoint_name+'_'+str(ind))
        #if ind % print_per_loops == 1:
        #    print('loss is: {}' .format(loss.item()))
        print('{} batch\'s loss is: {}' .format(ind, loss.item()))

def Predict(model, data, model_checkpoint_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model = loadModel(model,  model_checkpoint_name)

    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data)
        prob = 1/(1+np.exp(-1*out.to('cpu')))
    return prob



def saveModel(model, model_checkpoint_name):
    torch.save(model, model_checkpoint_name)

def loadModel(model, model_checkpoint_name):
    if os.path.exists(model_checkpoint_name):
        model = torch.load(model_checkpoint_name)
        model.eval()
        print('Load model successfully!')
        return model
    else:
        return model


