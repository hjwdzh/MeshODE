import sys
import os
import numpy as np
from time import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.graph_loss_layer import GraphLossLayer, Finalize
from layers.reverse_loss_layer import ReverseLossLayer
import pyDeform

source_path = sys.argv[1]
reference_path = sys.argv[2]
output_path = sys.argv[3]
rigidity = float(sys.argv[4])
src_V, src_F, src_E, src_to_graph, graph_V, graph_E\
	= pyDeform.LoadCadMesh(source_path)

tar_V, tar_F, tar_E, tar_to_graph, graph_V_tar, graph_E_tar\
	= pyDeform.LoadCadMesh(reference_path)

graph_deform = GraphLossLayer(graph_V, graph_E, tar_V, tar_F, rigidity)
param_id = graph_deform.param_id
reverse_deform = ReverseLossLayer()

graph_V = nn.Parameter(graph_V)
optimizer = optim.Adam([graph_V], lr=1e-3)

pyDeform.NormalizeByTemplate(graph_V_tar, param_id.tolist())
loss_src2tar, loss_tar2src = None, None
niter = 10000
prev_loss_src, prev_loss_tar = 1e30, 1e30
for it in range(0, niter):
	optimizer.zero_grad()	
	loss_src2tar = graph_deform(graph_V, graph_E)
	loss_tar2src = reverse_deform(graph_V, graph_V_tar)
	loss = loss_src2tar / graph_V.shape[0] + loss_tar2src / graph_V_tar.shape[0]
	loss.backward()
	optimizer.step()

	if it % 100 == 0:
		current_loss_src = np.sqrt(loss_src2tar.item() / graph_V.shape[0])
		current_loss_tar = np.sqrt(loss_tar2src.item() / graph_V_tar.shape[0])
		print('iter=%d, loss_src2tar=%.6f loss_tar2src=%.6f'
			%(it, np.sqrt(loss_src2tar.item() / graph_V.shape[0]),
			  np.sqrt(loss_tar2src.item() / graph_V_tar.shape[0])))
		if prev_loss_src - current_loss_src < 1e-6\
			and prev_loss_tar - current_loss_tar < 1e-6:
				break
		prev_loss_src, prev_loss_tar = current_loss_src, current_loss_tar

Finalize(src_V, src_F, src_E, src_to_graph, graph_V, 1, param_id)
pyDeform.SaveMesh(output_path, src_V, src_F)
