import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform

class RigidLossFunction(Function):
	@staticmethod
	def forward(ctx, src_V, src_F, param_id):
		pid = param_id.tolist()
		lossD = pyDeform.DistanceFieldLoss_forward(src_V, pid) * 0.5
		lossR = pyDeform.RigidEdgeLoss_forward(src_V, src_F, pid) * 0.5

		variables = [src_V, src_F, param_id]
		ctx.save_for_backward(*variables)

		return lossD.sum() + lossR.sum()

	@staticmethod
	def backward(ctx, grad_h):
		src_V = ctx.saved_variables[0]
		src_F = ctx.saved_variables[1]
		param_id = ctx.saved_variables[2].tolist()
		lossD_gradient = pyDeform.DistanceFieldLoss_backward(src_V, param_id)
		lossR_gradient = pyDeform.RigidEdgeLoss_backward(src_V, src_F, param_id)

		return grad_h*(lossD_gradient + lossR_gradient), None, None


class RigidLossLayer(nn.Module):
	def __init__(self, src_V, src_F, tar_V, tar_F):
		super(RigidLossLayer, self).__init__()
		
		self.param_id = torch.tensor(\
			pyDeform.InitializeDeformTemplate(tar_V, tar_F, 0, 64))

		pyDeform.NormalizeByTemplate(src_V, self.param_id.tolist())
		pyDeform.StoreRigidityInformation(src_V, src_F, self.param_id.tolist())

	def forward(self, src_V, src_F):
		return RigidLossFunction.apply(src_V, src_F, self.param_id)

def Finalize(src_V, param_id):
	pyDeform.DenormalizeByTemplate(src_V, param_id.tolist())