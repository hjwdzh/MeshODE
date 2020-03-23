import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform

class CadLossFunction(Function):
	@staticmethod
	def forward(ctx, src_V, src_F, src_E, param_id):
		lossD = pyDeform.DistanceFieldLoss_forward(src_V) * 0.5
		lossR = pyDeform.CadEdgeLoss_forward(src_V, src_F, src_E) * 0.5

		variables = [src_V, src_F, src_E, param_id]
		ctx.save_for_backward(*variables)

		return lossD.sum() + lossR.sum()

	@staticmethod
	def backward(ctx, grad_h):
		src_V = ctx.saved_variables[0]
		src_F = ctx.saved_variables[1]
		src_E = ctx.saved_variables[2]
		param_id = ctx.saved_variables[3].tolist()

		lossD_gradient = pyDeform.DistanceFieldLoss_backward(src_V, param_id)
		lossR_gradient = pyDeform.CadEdgeLoss_backward(\
			src_V, src_F, src_E, param_id)

		return grad_h*(lossD_gradient + lossR_gradient),\
			None, None, None


class CadLossLayer(nn.Module):
	def __init__(self, src_V, src_F, src_E, tar_V, tar_F):
		super(CadLossLayer, self).__init__()
		
		self.param_id = torch.tensor(\
			pyDeform.InitializeDeformTemplate(tar_V, tar_F, 0, 64))

		pyDeform.NormalizeByTemplate(src_V, self.param_id.tolist())
		pyDeform.StoreCadInformation(src_V, src_F, src_E,\
			self.param_id.tolist())

	def forward(self, src_V, src_F, src_E):
		return CadLossFunction.apply(src_V, src_F, src_E, self.param_id)

def Finalize(src_V, param_id):
	pyDeform.DenormalizeByTemplate(src_V, param_id.tolist())