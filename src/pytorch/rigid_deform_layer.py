import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform

class RigidDeformFunction(Function):
	@staticmethod
	def forward(ctx, src_V, src_F):
		lossD = pyDeform.DistanceFieldLoss_forward(src_V) * 0.5
		lossR = pyDeform.EdgeLoss_forward(src_V, src_F) * 0.5

		variables = [src_V, src_F]
		ctx.save_for_backward(*variables)

		return lossD.sum() + lossR.sum()

	@staticmethod
	def backward(ctx, grad_h):
		src_V = ctx.saved_variables[0]
		src_F = ctx.saved_variables[1]
		lossD_gradient = pyDeform.DistanceFieldLoss_backward(src_V)
		lossR_gradient = pyDeform.EdgeLoss_backward(src_V, src_F)

		return grad_h*(lossD_gradient + lossR_gradient), None


class RigidDeformLayer(nn.Module):
	def __init__(self, src_V, src_F, tar_V, tar_F):
		super(RigidDeformLayer, self).__init__()
		
		pyDeform.InitializeDeformTemplate(tar_V, tar_F, 0, 64);

		pyDeform.NormalizeByTemplate(src_V)
		pyDeform.StoreRigidityInformation(src_V, src_F)

	def forward(self, src_V, src_F):
		return RigidDeformFunction.apply(src_V, src_F)

