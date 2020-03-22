import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.graph_deform_layer import GraphDeformLayer, Finalize
import pyDeform

source_path = sys.argv[1]
reference_path = sys.argv[2]
output_path = sys.argv[3]
src_V, src_F, src_E, src_to_graph, graph_V, graph_E\
	= pyDeform.LoadCadMesh(source_path)

tar_V, tar_F = pyDeform.LoadMesh(reference_path)


graph_deform = GraphDeformLayer(graph_V, graph_E, tar_V, tar_F)
graph_V = nn.Parameter(graph_V)

optimizer = optim.Adam([graph_V], lr=1e-3)

niter = 10000
for it in range(0, niter):
	optimizer.zero_grad()	
	loss = graph_deform(graph_V, graph_E)
	loss.backward()
	optimizer.step()
	if it % 100 == 0:
		print('iter=%d, loss=%.6f'%(it, loss.item()))

Finalize(src_V, src_F, src_E, src_to_graph, graph_V)
pyDeform.SaveMesh(output_path, src_V, src_F)
