

import torch
import pyDeform
import numpy as np

tar_V, tar_F = pyDeform.LoadMesh('../data/reference.obj')
src_V, src_F = pyDeform.LoadMesh('../data/source.obj')

pyDeform.InitializeDeformTemplate(tar_V, tar_F, 1, 64);

src_V0 = src_V.clone()

pyDeform.NormalizeByTemplate(src_V)
pyDeform.StoreRigidityInformation(src_V, src_F)

#l1 = pyDeform.DistanceFieldLoss_forward(src_V)
#l2 = pyDeform.EdgeLoss_forward(src_V, src_F)

for it in range(10000):
	#lossD = pyDeform.EdgeLoss_forward(src_V, src_F)
	#lossD_gradient = pyDeform.EdgeLoss_backward(src_V, src_F)
	
	lossD = pyDeform.DistanceFieldLoss_forward(src_V) * 0.5
	lossR = pyDeform.EdgeLoss_forward(src_V, src_F) * 0.5

	lossD_gradient = pyDeform.DistanceFieldLoss_backward(src_V)
	lossR_gradient = pyDeform.EdgeLoss_backward(src_V, src_F)

	loss = lossD.sum() + lossR.sum()
	loss_gradient = lossD_gradient + lossR_gradient

	src_V -= loss_gradient * float(1e-3)
	print('loss = %.6f\n'%((loss).sum().item()))

pyDeform.DenormalizeByTemplate(src_V)

pyDeform.SaveMesh("output.obj", src_V, src_F)