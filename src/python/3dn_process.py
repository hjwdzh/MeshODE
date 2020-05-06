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
from layers.neuralode_fast import NeuralODE
import pyDeform

import torch
import numpy as np
from time import time

from scipy.spatial import cKDTree
'''
source_path = sys.argv[1]
reference_path = sys.argv[2]
output_source_path = sys.argv[3]
output_source_pts_path = sys.argv[4]
output_target_pts_path = sys.argv[5]

V1, F1, E1, V2G1, GV1, GE1 = pyDeform.LoadCadMesh(source_path)
V2, F2, E2, V2G2, GV2, GE2 = pyDeform.LoadCadMesh(reference_path)

pyDeform.SaveMesh(output_source_path, V1, F1)
choices = np.random.choice(V1.shape[0], 2048)
src_points = V1[choices, :]
np.savetxt(output_source_pts_path, src_points)

choices = np.random.choice(V2.shape[0], 2048)
tar_points = V2[choices, :]
np.savetxt(output_target_pts_path, tar_points)

'''
t1 = time()
source_path = sys.argv[1]
source_txt = sys.argv[2]
deform_txt = sys.argv[3]
output_txt = sys.argv[4]
src_V, src_F = pyDeform.LoadMesh(source_path)

sample_src_V = np.loadtxt(source_txt)
deform_src_V = np.loadtxt(deform_txt)

print(sample_src_V.shape, deform_src_V.shape)
offset = deform_src_V - sample_src_V

src_V_numpy = src_V.data.cpu().numpy()

tree = cKDTree(sample_src_V)
dd, ii = tree.query(src_V_numpy, k=3, n_jobs=1)

for i in range(src_V.shape[0]):
	w = 1.0 / (dd[i] + 1e-6)
	w = w / np.sum(w)
	
	t = w[0] * offset[ii[i][0]] + w[1] * offset[ii[i][1]] + w[2] * offset[ii[i][2]]
	src_V_numpy[i] += t

src_V = torch.from_numpy(src_V_numpy)

pyDeform.SaveMesh(output_txt, src_V, src_F)
t2 = time()
print(t2 - t1)