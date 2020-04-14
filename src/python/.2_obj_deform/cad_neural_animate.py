import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.graph_loss2_layer import GraphLoss2Layer, Finalize
from layers.reverse_loss_layer import ReverseLossLayer
from layers.maf import MAF
from layers.neuralode import NeuralODE
import pyDeform

import torch
import numpy as np
from time import time

source_path = sys.argv[1]
reference_path = sys.argv[2]
output_path = sys.argv[3]
rigidity = float(sys.argv[4])

if len(sys.argv) > 5:
	device = torch.device(sys.argv[5])
else:
	device = torch.device('cpu')


V1, F1, E1, V2G1, GV1, GE1 = pyDeform.LoadCadMesh(source_path)
V2, F2, E2, V2G2, GV2, GE2 = pyDeform.LoadCadMesh(reference_path)

graph_loss = GraphLoss2Layer(V1,F1,GV1,GE1,V2,F2,GV2,GE2,rigidity,device)
param_id1 = graph_loss.param_id1
param_id2 = graph_loss.param_id2

reverse_loss = ReverseLossLayer()

#func = MAF(5, 3, 256, 1, None, 'relu', 'sequential', batch_norm=False)
#func = func.to(device)

if os.path.exists('models/func.ckpt'):
	checkpoint = torch.load('models/func.ckpt')

	optimizer = checkpoint['optim']
	func = checkpoint['func']
	func.func = func.func.to(device)
else:
	func = NeuralODE(device)
	optimizer = optim.Adam(func.parameters(), lr=1e-3)

GV1_origin = GV1.clone()
GV2_origin = GV2.clone()

niter = 0

GV1_device = GV1.to(device)
GV2_device = GV2.to(device)
loss_min = 1e30
for it in range(0, niter):
	optimizer.zero_grad()

	GV1_deformed = func.forward(GV1_device)
	GV2_deformed = func.inverse(GV2_device)

	loss1_forward = graph_loss(GV1_deformed, GE1, GV2, GE2, 0)
	loss1_backward = reverse_loss(GV1_deformed, GV2_origin, device)

	loss2_forward = graph_loss(GV1, GE1, GV2_deformed, GE2, 1)
	loss2_backward = reverse_loss(GV2_deformed, GV1_origin, device)

	loss = loss1_forward + loss1_backward + loss2_forward + loss2_backward

	loss.backward()
	optimizer.step()

	if it % 100 == 0 or True:
		print('iter=%d, loss1_forward=%.6f loss1_backward=%.6f loss2_forward=%.6f loss2_backward=%.6f'
			%(it, np.sqrt(loss1_forward.item() / GV1.shape[0]),
				np.sqrt(loss1_backward.item() / GV2.shape[0]),
				np.sqrt(loss2_forward.item() / GV2.shape[0]),
				np.sqrt(loss2_backward.item() / GV1.shape[0])))

		current_loss = loss.item()

torch.save({'func':func, 'optim':optimizer}, 'models/func.ckpt')
GV1_deformed = func.forward(GV1_device)
GV1_deformed = torch.from_numpy(GV1_deformed.data.cpu().numpy())
V1_copy = V1.clone()
#Finalize(V1_copy, F1, E1, V2G1, GV1_deformed, 1.0, param_id2)
pyDeform.NormalizeByTemplate(V1_copy, param_id1.tolist())
V1_origin = V1_copy.clone()


for i in range(26):
	if i != 25:
		continue
	V1_deform = V1_origin.clone()
	V1_copy = V1_origin.clone().to(device)
	if i != 0:
		V1_copy = func.integrate(V1_copy, 0, i * 0.04, device)
	V1_copy = torch.from_numpy(V1_copy.data.cpu().numpy())

	src_to_src = torch.from_numpy(np.array([i for i in range(V1_origin.shape[0])]).astype('int32'))
	E1 = np.zeros((F1.shape[0] * 3, 2), dtype='int32')
	F1_numpy = F1.numpy()
	E1[:F1.shape[0],:] = F1_numpy[:,0:2]
	E1[F1.shape[0]:F1.shape[0]*2,:] = F1_numpy[:,1:3]
	E1[F1.shape[0]*2:,0] = F1_numpy[:,2]
	E1[F1.shape[0]*2:,1] = F1_numpy[:,0]
	E1 = torch.from_numpy(E1)

	pyDeform.SolveLinear(V1_deform, F1, E1, src_to_src, V1_copy, 1, 1)
	pyDeform.DenormalizeByTemplate(V1_deform, param_id2.tolist())
	pyDeform.SaveMesh('result/src-%02d.obj'%(i), V1_deform, F1)

exit(0)
V2_copy = V2.clone()
pyDeform.NormalizeByTemplate(V2_copy, param_id2.tolist())
V2_origin = V2_copy.clone()

for i in range(26):
	V2_deform = V2_origin.clone()
	V2_copy = V2_origin.clone().to(device)
	if i != 25:
		V2_copy = func.integrate(V2_copy, 1, i * 0.04, device)
	V2_copy = torch.from_numpy(V2_copy.data.cpu().numpy())

	src_to_src = torch.from_numpy(np.array([i for i in range(V2_origin.shape[0])]).astype('int32'))

	pyDeform.SolveLinear(V2_deform, F2, E2, src_to_src, V2_copy, 1, 1)
	pyDeform.DenormalizeByTemplate(V2_deform, param_id2.tolist())
	pyDeform.SaveMesh('result/tar-%02d.obj'%(i), V2_deform, F2)
