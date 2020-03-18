import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform

class CadDeformFunction(Function):
	@staticmethod
	def forward(ctx, src_V, src_F, src_E):
		lossD = pyDeform.DistanceFieldLoss_forward(src_V) * 0.5
		lossR = pyDeform.CadEdgeLoss_forward(src_V, src_F, src_E) * 0.5

		variables = [src_V, src_F, src_E]
		ctx.save_for_backward(*variables)

		return lossD.sum() + lossR.sum()

	@staticmethod
	def backward(ctx, grad_h):
		src_V = ctx.saved_variables[0]
		src_F = ctx.saved_variables[1]
		src_E = ctx.saved_variables[2]

		lossD_gradient = pyDeform.DistanceFieldLoss_backward(src_V)
		lossR_gradient = pyDeform.CadEdgeLoss_backward(src_V, src_F, src_E)

		return grad_h*(lossD_gradient + lossR_gradient), None, None


class CadDeformLayer(nn.Module):
	def __init__(self, src_V, src_F, src_E, tar_V, tar_F):
		super(CadDeformLayer, self).__init__()
		
		pyDeform.InitializeDeformTemplate(tar_V, tar_F, 0, 64);

		pyDeform.NormalizeByTemplate(src_V)
		pyDeform.StoreCadInformation(src_V, src_F, src_E)

	def forward(self, src_V, src_F, src_E):
		return CadDeformFunction.apply(src_V, src_F, src_E)

def Finalize(src_V):
	pyDeform.DenormalizeByTemplate(src_V)