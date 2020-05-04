import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.reverse_loss_layer import ReverseLossLayer
from layers.rigid_loss_layer import RigidLossLayer, Finalize
import pyDeform

source_path = sys.argv[1]
reference_path = sys.argv[2]
output_path = sys.argv[3]
src_V, src_F = pyDeform.LoadMesh(source_path)
tar_V, tar_F = pyDeform.LoadMesh(reference_path)

rigid_deform = RigidLossLayer(src_V, src_F, tar_V, tar_F)
reverse_deform = ReverseLossLayer()

param_id = rigid_deform.param_id
src_V = nn.Parameter(src_V)

optimizer = optim.Adam([src_V], lr=1e-3)
pyDeform.NormalizeByTemplate(tar_V, param_id.tolist())
niter = 10000
prev_loss = 1e30
for it in range(0, niter):
	optimizer.zero_grad()	
	loss_src2tar = rigid_deform(src_V, src_F)
	loss_tar2src = reverse_deform(src_V, tar_V)
	loss = loss_src2tar * 10 + loss_tar2src
	loss.backward()
	optimizer.step()
	if it % 100 == 0:
		l = loss.item()
		print('iter=%d loss=%.6f loss_tar2src=%.6f'%(it, l, np.sqrt((loss_tar2src/tar_V.shape[0]).item())))
		if l > prev_loss:
			break
		prev_loss = l

Finalize(src_V, param_id)
pyDeform.SaveMesh(output_path, src_V, src_F)
#/orion/downloads/deformation_aware_embedding/scan2cad/version3/03001627/7729a6ad96985f4ed1ccbd5d84e5bc86_scan_0021.obj