import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.rigid_deform_layer import RigidDeformLayer, Finalize
import pyDeform

source_path = sys.argv[1]
reference_path = sys.argv[2]
output_path = sys.argv[3]
src_V, src_F = pyDeform.LoadMesh(source_path)
tar_V, tar_F = pyDeform.LoadMesh(reference_path)

rigid_deform = RigidDeformLayer(src_V, src_F, tar_V, tar_F)
src_V = nn.Parameter(src_V)

optimizer = optim.Adam([src_V], lr=1e-3)

niter = 10000
for it in range(0, niter):
	optimizer.zero_grad()	
	loss = rigid_deform(src_V, src_F)
	loss.backward()
	optimizer.step()
	if it % 100 == 0:
		print('iter=%d loss=%.6f'%(it, loss.item()))

Finalize(src_V)
pyDeform.SaveMesh(output_path, src_V, src_F)
