import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform

class GraphLossFunction(Function):
	@staticmethod
	def forward(ctx, src_V, src_E, rigidity2, param_id):
		pid = param_id.tolist()
		lossD = pyDeform.DistanceFieldLoss_forward(src_V, pid) * 0.5
		lossR = pyDeform.GraphEdgeLoss_forward(src_V, src_E, pid) * 0.5

		variables = [src_V, src_E, rigidity2, param_id]
		ctx.save_for_backward(*variables)

		return lossD.sum() + lossR.sum() * rigidity2.tolist()

	@staticmethod
	def backward(ctx, grad_h):
		src_V = ctx.saved_variables[0]
		src_E = ctx.saved_variables[1]
		rigidity2 = ctx.saved_variables[2]
		param_id = ctx.saved_variables[3].tolist()

		lossD_gradient = pyDeform.DistanceFieldLoss_backward(src_V, param_id)
		lossR_gradient = pyDeform.GraphEdgeLoss_backward(src_V, src_E, param_id)

		return grad_h*(lossD_gradient + lossR_gradient*rigidity2.tolist()),\
			None, None, None


class GraphLossLayer(nn.Module):
	def __init__(self, src_V, src_E, tar_V, tar_F, rigidity):
		super(GraphLossLayer, self).__init__()

		self.param_id = torch.tensor(\
			pyDeform.InitializeDeformTemplate(tar_V, tar_F, 0, 64))

		pyDeform.NormalizeByTemplate(src_V, self.param_id.tolist())
		pyDeform.StoreGraphInformation(src_V, src_E, self.param_id.tolist())
		self.rigidity2 = torch.tensor(rigidity * rigidity)

	def forward(self, src_V, src_E):
		return GraphLossFunction.apply(src_V, src_E,\
			self.rigidity2, self.param_id)

def Finalize(src_V, src_F, src_E, src_to_graph, graph_V, rigidity, param_id):
	pyDeform.NormalizeByTemplate(src_V, param_id.tolist())
	pyDeform.SolveLinear(src_V, src_F, src_E, src_to_graph, graph_V, rigidity)
	pyDeform.DenormalizeByTemplate(src_V, param_id.tolist())