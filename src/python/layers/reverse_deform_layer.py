import math
from torch import nn
from torch.autograd import Function
import torch

from sklearn.neighbors import NearestNeighbors

class ReverseDeformLayer(nn.Module):
	def __init__(self):
		super(ReverseDeformLayer, self).__init__()
		
	def forward(self, src_V, tar_V):
		src_V_numpy = src_V.data.numpy()
		tar_V_numpy = tar_V.data.numpy()
		nbrs = NearestNeighbors(n_neighbors=1,
			algorithm='ball_tree').fit(src_V_numpy)
		distances, indices = nbrs.kneighbors(tar_V_numpy)
		
		src_V_c = src_V[indices[:,0]]
		loss = src_V_c - tar_V
		loss = 0.5 * (loss * loss).sum()

		return loss