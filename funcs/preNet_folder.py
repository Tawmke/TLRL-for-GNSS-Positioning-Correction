import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def loadNet():
    data_folder = './newNet/'
    pkl_route = data_folder + 'new.pkl'
    with open(pkl_route, "rb") as file:
        params = pickle.load(file)
    file.close()
    net = Net(params)
    return net

def loadNet_preNet(filename):
    data_folder = './newNet/'
    pkl_route = data_folder + 'new.pkl'
    with open(pkl_route, "rb") as file:
        params = pickle.load(file)
    file.close()
    net = Net(params)
    return net

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 32, bias=True).double()
        # self.fc.data = torch.DoubleTensor(params)
        self.fc.weight.data = torch.DoubleTensor(params['PreNet'].T)
        self.fc.bias.data = torch.DoubleTensor(params['Bias'].reshape(-1).T)
        pass

    def forward(self, x):
        res = x.detach()
        res[1] = torch.tanh(10 * res[1]) * 0.7
        res = self.fc(res)
        res = F.relu(res)
        return res


if __name__ == '__main__':
    torch.set_printoptions(precision=8)
    net = loadNet()
    a = torch.from_numpy(np.random.rand(2))
    print(a)
    input = torch.DoubleTensor(a)
    print(net(input).detach().numpy().dtype)
    print(a)
    print(np.float64 == np.double)
