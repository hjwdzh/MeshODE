import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from rigid_deform_layer import RigidDeformLayer
import pyDeform

tar_V, tar_F = pyDeform.LoadMesh('../data/reference.obj')
src_V, src_F = pyDeform.LoadMesh('../data/source.obj')

rigid_deform = RigidDeformLayer(src_V, src_F, tar_V, tar_F)
src_V = nn.Parameter(src_V)

optimizer = optim.SGD([src_V], lr=1e-3)

niter = 10000
for _ in range(0, niter):
	optimizer.zero_grad()	
	loss = rigid_deform(src_V, src_F)
	loss.backward()
	optimizer.step()
	print('loss=%.6f'%(loss.item()))

