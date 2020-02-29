

import torch
import pyDeform
import numpy as np

print('step1')
src_V, src_F = pyDeform.LoadMesh('../data/source.obj')
print('step2')
tar_V, tar_F = pyDeform.LoadMesh('../data/reference.obj')
print('step3', src_V.shape, tar_V.shape)

pyDeform.InitializeDeformTemplate(tar_V, tar_F, 1, 64, 1.0);

print(src_V[0])
pyDeform.NormalizeByTemplate(src_V)
print(src_V[0])
