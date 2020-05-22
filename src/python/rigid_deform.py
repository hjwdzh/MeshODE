import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.rigid_loss_layer import RigidLossLayer, Finalize
import pyDeform

import argparse

parser = argparse.ArgumentParser(description='Rigid Deformation.')
parser.add_argument('--source', default='../data/source.obj')
parser.add_argument('--target', default='../data/target.obj')
parser.add_argument('--output', default='./output.obj')

args = parser.parse_args()

source_path = args.source
reference_path = args.target
output_path = args.output
src_V, src_F = pyDeform.LoadMesh(source_path)
tar_V, tar_F = pyDeform.LoadMesh(reference_path)

rigid_deform = RigidLossLayer(src_V, src_F, tar_V, tar_F)
param_id = rigid_deform.param_id
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

Finalize(src_V, param_id)
pyDeform.SaveMesh(output_path, src_V, src_F)
