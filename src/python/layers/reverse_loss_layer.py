import math
from torch import nn
from torch.autograd import Function
import torch

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from time import time

class ReverseLossLayer(nn.Module):
	def __init__(self):
		super(ReverseLossLayer, self).__init__()
		
	def forward(self, src_V, tar_V, device=torch.device('cpu')):
		src_V_numpy = src_V.data.cpu().numpy()
		tar_V_numpy = tar_V.data.cpu().numpy()

		tree = cKDTree(src_V_numpy)
		dd, ii = tree.query(tar_V_numpy, k=1, n_jobs=1)

		src_V_c = src_V[ii]
		loss = src_V_c - tar_V.to(device)

		loss = 0.5 * (loss * loss).sum()

		return loss