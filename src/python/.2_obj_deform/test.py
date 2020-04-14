import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

#if args.adjoint:
#    from torchdiffeq import odeint_adjoint as odeint
#else:
from torchdiffeq import odeint

data = np.load('deform.npz')
X0 = data['src']
X1 = data['tar']
#X0 = np.reshape(X0, (X0.shape[0], 1, X0.shape[1])).astype('float32')
#X1 = np.reshape(X1, (X1.shape[0], 1, X1.shape[1])).astype('float32')

X1 = np.array([X0, X1])

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        m = 50
        nlin = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(4, m),
            nlin,
            nn.Linear(m, m),
            nlin,
            nn.Linear(m, 3),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=8e-2)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        new_t = t.repeat(y.shape[0],1)
        yt = torch.cat((y,new_t), 1)
        return self.net(yt-0.5)


timing = np.array([0, 1]).astype('float32')
timing_inv = np.array([1, 0]).astype('float32')
X0, X1, timing, timing_inv = torch.from_numpy(X0), torch.from_numpy(X1), torch.from_numpy(timing), torch.from_numpy(timing_inv)

func = ODEFunc()
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

for itr in range(0, 10000):
    optimizer.zero_grad()
    
    pred_y = odeint(func, X0, timing)
    loss = torch.mean(torch.abs(pred_y[1] - X1[1]))
    loss.backward()

    inverse_y = odeint(func, pred_y[1], timing_inv)

    print('iter = %d, loss = %.6f'%(itr, loss.item()))
    optimizer.step()
