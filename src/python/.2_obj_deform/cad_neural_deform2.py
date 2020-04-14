import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.graph_loss2_layer import GraphLoss2Layer, Finalize
from layers.chamfer_layer import ChamferLayer, ChamferLoss
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

# chamfer_loss = ChamferLayer()
chamfer_loss = ChamferLoss(reduction='sum')

#func = MAF(5, 3, 256, 1, None, 'relu', 'sequential', batch_norm=False)
#func = func.to(device)
func = NeuralODE(device)

optimizer = optim.Adam(func.parameters(), lr=1e-3)
GV1_origin = GV1.clone()
GV2_origin = GV2.clone()

niter = 1000

GV1_device = GV1.to(device)
GV2_device = GV2.to(device)
loss_min = 1e30
for it in range(0, niter):
    optimizer.zero_grad()

    GV1_deformed = func.forward(GV1_device)
    GV2_deformed = func.inverse(GV2_device)

#     loss1_forward = graph_loss(GV1_deformed, GE1, GV2, GE2, 0)
#     loss1_forward = chamfer_loss(GV2_device, GV1_deformed.cpu(), device)
#     loss1_backward = chamfer_loss(GV1_deformed, GV2_origin, device)
    loss1_forward, loss1_backward, loss1_sum = chamfer_loss(GV1_deformed.unsqueeze(0), GV2_device.unsqueeze(0))
    

#     loss2_forward = graph_loss(GV1, GE1, GV2_deformed, GE2, 1)
#     loss2_forward = chamfer_loss(GV1_device, GV2_deformed.cpu(), device)
#     loss2_backward = chamfer_loss(GV2_deformed, GV1_origin, device)
    loss2_forward, loss2_backward, loss2_sum = chamfer_loss(GV1_device.unsqueeze(0), GV2_deformed.unsqueeze(0))

    loss = loss1_sum = loss2_sum

    loss.backward()
    optimizer.step()

    if it % 100 == 0 or True:
        print('iter=%d, loss1_forward=%.6f loss1_backward=%.6f loss2_forward=%.6f loss2_backward=%.6f'
            %(it, np.sqrt(loss1_forward.item() / GV1.shape[0]),
                np.sqrt(loss1_backward.item() / GV2.shape[0]),
                np.sqrt(loss2_forward.item() / GV2.shape[0]),
                np.sqrt(loss2_backward.item() / GV1.shape[0])))

        current_loss = loss.item()

GV1_deformed = func.forward(GV1_device)
GV1_deformed = torch.from_numpy(GV1_deformed.data.cpu().numpy())
V1_copy = V1.clone()
#Finalize(V1_copy, F1, E1, V2G1, GV1_deformed, 1.0, param_id2)

pyDeform.NormalizeByTemplate(V1_copy, param_id1.tolist())
V1_origin = V1_copy.clone()

V1_copy = V1_copy.to(device)
V1_copy = func.forward(V1_copy)
V1_copy = torch.from_numpy(V1_copy.data.cpu().numpy())

src_to_src = torch.from_numpy(np.array([i for i in range(V1_origin.shape[0])]).astype('int32'))

pyDeform.SolveLinear(V1_origin, F1, E1, src_to_src, V1_copy, 1, 1)
pyDeform.DenormalizeByTemplate(V1_origin, param_id2.tolist())
pyDeform.SaveMesh(output_path, V1_origin, F1)
