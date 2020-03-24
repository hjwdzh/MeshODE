import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.graph_loss_layer import GraphLossLayer, Finalize
import pyDeform

import torch
import numpy as np

source_path = sys.argv[1]
reference_path = sys.argv[2]
output_path = sys.argv[3]
rigidity = float(sys.argv[4])

src_V, src_F, src_E, src_to_graph, graph_V_src, graph_E_src\
	= pyDeform.LoadCadMesh(source_path)

tar_V, tar_F, tar_E, tar_to_graph, graph_V_tar, graph_E_tar\
	= pyDeform.LoadCadMesh(reference_path)

graph_loss_src2tar = GraphLossLayer(\
	graph_V_src, graph_E_src, tar_V, tar_F, rigidity)

param_src2tar = graph_loss_src2tar.param_id

graph_loss_tar2src = GraphLossLayer(\
	graph_V_tar, graph_E_tar, src_V, src_F, rigidity)

param_tar2src = graph_loss_tar2src.param_id

from torchdiffeq import odeint

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
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        new_t = t.repeat(y.shape[0],1)
        yt = torch.cat((y,new_t), 1)
        return self.net(yt-0.5)

timing = torch.from_numpy(np.array([0, 1]).astype('float32'))
timing_inv = torch.from_numpy(np.array([1, 0]).astype('float32'))

func = ODEFunc()

optimizer = optim.Adam(func.parameters(), lr=1e-3)

niter = 500

for it in range(0, niter):
	optimizer.zero_grad()

	graph_V_src_deform = odeint(func, graph_V_src, timing)[1]
	loss_src2tar = graph_loss_src2tar(graph_V_src_deform, graph_E_src)
	graph_V_tar_deform = odeint(func, graph_V_tar, timing_inv)[1]
	loss_tar2src = graph_loss_tar2src(graph_V_tar_deform, graph_E_tar)

	loss = loss_src2tar + loss_tar2src

	loss.backward()

	optimizer.step()
	if it % 100 == 0 or True:
		print('iter=%d, loss_src2tar=%.6f, loss_tar2src=%.6f'%(it, loss_src2tar.item(), loss_tar2src.item()))


graph_V_src = torch.from_numpy(odeint(func, graph_V_src, timing)[1].data.numpy())
Finalize(src_V, src_F, src_E, src_to_graph, graph_V_src, rigidity, param_src2tar)
pyDeform.SaveMesh(output_path, src_V, src_F)
