import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 'layers')

from torch import nn
import torch.optim as optim
from torch.autograd import Function

from layers.chamfer_layer import ChamferDist, ChamferDistKDTree
from layers.deformation_layer import NeuralFlowDeformer
import pyDeform
from layers.graph_loss2_layer import GraphLoss2Layer, Finalize
from layers.pointnet_model import PointNetEncoder

import torch
import numpy as np
from time import time
import trimesh
from glob import glob


files = sorted(glob("/home/maxjiang/codes/ShapeDeform/data/03001627/*/*.ply"))
m1 = trimesh.load(files[1])
m2 = trimesh.load(files[6])
m3 = trimesh.load(files[7])
device = torch.device('cuda:0')

chamfer_dist = ChamferDistKDTree(reduction='mean', njobs=1)
criterion = torch.nn.MSELoss()

latent_size = 3

deformer = NeuralFlowDeformer(latent_size=latent_size, f_nlayers=6, f_width=100, s_nlayers=2, s_width=5, method='rk4', nonlinearity='leakyrelu', arch='imnet', device=device)
encoder = PointNetEncoder(nf=16, output_features=latent_size).to(device)

optimizer = optim.Adam(list(deformer.parameters)+list(encoder.parameters()), lr=1e-3)

niter = 1000
npts = 5000

V1 = torch.tensor(m1.vertices.astype(np.float32)).to(device)  # .unsqueeze(0)
V2 = torch.tensor(m2.vertices.astype(np.float32)).to(device)  # .unsqueeze(0)
V3 = torch.tensor(m3.vertices.astype(np.float32)).to(device)  # .unsqueeze(0)
# V1_latent = torch.tensor(np.array([[1., 0., 0.]], dtype=np.float32)).to(device)
# V2_latent = torch.tensor(np.array([[0., 1., 0.]], dtype=np.float32)).to(device)
# V3_latent = torch.tensor(np.array([[0., 0., 1.]], dtype=np.float32)).to(device)
# batch_latent_src = torch.cat([V1_latent, V3_latent, V1_latent, V2_latent, V2_latent, V3_latent], dim=0)
# batch_latent_tar = torch.cat([V3_latent, V1_latent, V2_latent, V1_latent, V3_latent, V2_latent], dim=0)

loss_min = 1e30
tic = time()
encoder.train()

for it in range(0, niter):
    optimizer.zero_grad()

    seq1 = torch.randperm(V1.shape[0], device=device)[:npts]
    seq2 = torch.randperm(V2.shape[0], device=device)[:npts]
    seq3 = torch.randperm(V3.shape[0], device=device)[:npts]
    V1_samp = V1[seq1]
    V2_samp = V2[seq2]
    V3_samp = V3[seq3]

    V_src = torch.stack([V1_samp, V3_samp, V1_samp, V2_samp, V2_samp, V3_samp], dim=0)  # [batch, npoints, 3]
    V_tar = torch.stack([V3_samp, V1_samp, V2_samp, V1_samp, V3_samp, V2_samp], dim=0)  # [batch, npoints, 3]
    
    batch_latent_src = encoder(V_src)
    batch_latent_tar = encoder(V_tar)
    
    V_deform = deformer.forward(batch_latent_src, batch_latent_tar, V_src)
    

    _, _, dist = chamfer_dist(V_deform, V_tar)

    loss = criterion(dist, torch.zeros_like(dist))

    loss.backward()
    optimizer.step()

    if it % 100 == 0 or True:
        print(f'iter={it}, loss={np.sqrt(loss.item())}')

toc = time()
print("Time for {} iters: {:.4f} s".format(niter, toc-tic))

# save deformed mesh
encoder.eval()
with torch.no_grad():
    V1_latent = encoder(V1.unsqueeze(0))
    V2_latent = encoder(V2.unsqueeze(0))
    V3_latent = encoder(V3.unsqueeze(0))

    V1_2 = deformer.forward(V1_latent, V2_latent, V1.unsqueeze(0)).detach().cpu().numpy()[0]
    V2_1 = deformer.forward(V2_latent, V1_latent, V2.unsqueeze(0)).detach().cpu().numpy()[0]
    V1_3 = deformer.forward(V1_latent, V3_latent, V1.unsqueeze(0)).detach().cpu().numpy()[0]
    V3_1 = deformer.forward(V3_latent, V1_latent, V3.unsqueeze(0)).detach().cpu().numpy()[0]
    V2_3 = deformer.forward(V2_latent, V3_latent, V2.unsqueeze(0)).detach().cpu().numpy()[0]
    V3_2 = deformer.forward(V3_latent, V2_latent, V3.unsqueeze(0)).detach().cpu().numpy()[0]
trimesh.Trimesh(V1_2, m1.faces).export('/home/maxjiang/codes/ShapeDeform/data/output_1_2.obj')
trimesh.Trimesh(V2_1, m2.faces).export('/home/maxjiang/codes/ShapeDeform/data/output_2_1.obj')
trimesh.Trimesh(V1_3, m1.faces).export('/home/maxjiang/codes/ShapeDeform/data/output_1_3.obj')
trimesh.Trimesh(V3_1, m3.faces).export('/home/maxjiang/codes/ShapeDeform/data/output_3_1.obj')
trimesh.Trimesh(V2_3, m2.faces).export('/home/maxjiang/codes/ShapeDeform/data/output_2_3.obj')
trimesh.Trimesh(V3_2, m3.faces).export('/home/maxjiang/codes/ShapeDeform/data/output_3_2.obj')

m1.export('/home/maxjiang/codes/ShapeDeform/data/output_1.obj')
m2.export('/home/maxjiang/codes/ShapeDeform/data/output_2.obj')
m3.export('/home/maxjiang/codes/ShapeDeform/data/output_3.obj')