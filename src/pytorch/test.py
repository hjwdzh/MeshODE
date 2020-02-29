

import torch
import pyDeform
import numpy as np

src_V, src_F = pyDeform.LoadMesh('../data/source.obj')
tar_V, tar_F = pyDeform.LoadMesh('../data/reference.obj')

pyDeform.InitializeDeformTemplate(tar_V, tar_F, 1, 64, 1.0);

src_V0 = src_V.clone()

pyDeform.NormalizeByTemplate(src_V)


for it in range(10000):
	lossD = pyDeform.RigidDeform_forward(src_V)
	lossD_gradient = pyDeform.RigidDeform_backward(src_V)
	src_V -= lossD_gradient * float(1e-4)
	print('loss = %.6f\n'%((lossD).mean().item()))

pyDeform.DenormalizeByTemplate(src_V)

