import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from cad_deform_layer import CadDeformLayer, Finalize
import pyDeform

source_path = sys.argv[1]
reference_path = sys.argv[2]
output_path = sys.argv[3]
src_V, src_F, src_E = pyDeform.LoadCadMesh(source_path)
tar_V, tar_F = pyDeform.LoadMesh(reference_path)

cad_deform = CadDeformLayer(src_V, src_F, src_E, tar_V, tar_F)
src_V = nn.Parameter(src_V)

optimizer = optim.Adam([src_V], lr=1e-3)

niter = 10000
for _ in range(0, niter):
	optimizer.zero_grad()	
	loss = cad_deform(src_V, src_F, src_E)
	loss.backward()
	optimizer.step()
	print('loss=%.6f'%(loss.item()))
	if loss.item() < 0.6:
		break

Finalize(src_V)
pyDeform.SaveMesh(output_path, src_V, src_F)
