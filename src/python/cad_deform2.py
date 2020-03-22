import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.graph_deform_layer import GraphDeformLayer, Finalize
from layers.reverse_deform_layer import ReverseDeformLayer
import pyDeform

source_path = sys.argv[1]
reference_path = sys.argv[2]
output_path = sys.argv[3]
src_V, src_F, src_E, src_to_graph, graph_V, graph_E\
	= pyDeform.LoadCadMesh(source_path)

tar_V, tar_F, tar_E, _, _, _\
	= pyDeform.LoadCadMesh(reference_path)

graph_deform = GraphDeformLayer(graph_V, graph_E, tar_V, tar_F)
reverse_deform = ReverseDeformLayer()
pyDeform.NormalizeByTemplate(tar_V)

graph_V = nn.Parameter(graph_V)
optimizer = optim.Adam([graph_V], lr=1e-3)

niter = 10000
for it in range(0, niter):
	optimizer.zero_grad()	
	loss_src2tar = graph_deform(graph_V, graph_E)
	loss_tar2src = reverse_deform(graph_V, tar_V)

	loss = loss_src2tar / graph_V.shape[0] + loss_tar2src / tar_V.shape[0]
	loss.backward()

	optimizer.step()

	if it % 100 == 0:
		print('iter=%d, loss_src2tar=%.6f loss_tar2src=%.6f'
			%(it, loss_src2tar.item(), loss_tar2src.item()))

Finalize(src_V, src_F, src_E, src_to_graph, graph_V)
pyDeform.SaveMesh(output_path, src_V, src_F)
