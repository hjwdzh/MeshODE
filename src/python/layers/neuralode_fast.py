import torch
from torch import nn
from torchdiffeq import odeint

import numpy as np

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        m = 50
        nlin = nn.LeakyReLu()
        self.net = nn.Sequential(
            nn.Linear(4, m),
            nlin,
            nn.Linear(m, m),
            nlin,
            nn.Linear(m, m),
            nlin,
            nn.Linear(m, 3),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        new_t = t.repeat(y.shape[0],1)
        yt = torch.cat((y,new_t), 1)
        res = self.net(yt - 0.5)
        return res
        #return self.net(yt-0.5)

class NeuralODE():
    def __init__(self, device=torch.device('cpu')):
        super(NeuralODE, self).__init__()
        self.timing = torch.from_numpy(np.array([0, 1]).astype('float32'))
        self.timing_inv = torch.from_numpy(np.array([1, 0]).astype('float32'))
        self.timing = self.timing.to(device)
        self.timing_inv = self.timing_inv.to(device)

        self.func = ODEFunc()
        self.func = self.func.to(device)
        self.device = device

    def to_device(self, device):
        self.func = self.func.to(device)
        self.timing = self.timing.to(device)
        self.timing_inv = self.timing_inv.to(device)
        self.device = device
        
    def parameters(self):
        return self.func.parameters()

    def forward(self, u):
        y = odeint(self.func, u, self.timing)[1]
        return y

    def inverse(self, u):
        return odeint(self.func, u, self.timing_inv)[1]

    def integrate(self, u, t1, t2, device):
        new_time = torch.from_numpy(np.array([t1,t2]).astype('float32')).to(device)
        return odeint(self.func, u, new_time, method="rk4", rtol=1e-4, atol=1e-4)[1]