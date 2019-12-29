import torch
import torch.nn as nn

class lstm(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=1,num_layer=2, seq_len = 50):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size*seq_len,output_size), 
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = x.permute(1, 0, 2)# #seq_len * batch * feature
        x, (h, c)= self.layer1(x)
        s,b,h = x.size()
        x = x.permute(1, 0, 2).contiguous()#.contiguous() #batch * seq_len * feature
        x = x.view(b,-1)
        x = self.layer2(x)
        return x

if __name__ == '__main__':
    seq_len = 10
    input_feature = 20
    hidden_size = 15
    n = lstm(input_feature, hidden_size, 1, 2, seq_len)
    data = torch.randn(5, 10, 20)
    out = n(data)
    print(out.shape)