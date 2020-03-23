import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform

rigidity2 = 1
class GraphDeformFunction(Function):
	@staticmethod
	def forward(ctx, src_V, src_E):
		global rigidity2
		lossD = pyDeform.DistanceFieldLoss_forward(src_V) * 0.5
		lossR = pyDeform.GraphEdgeLoss_forward(src_V, src_E) * 0.5

		variables = [src_V, src_E]
		ctx.save_for_backward(*variables)

		return lossD.sum() + lossR.sum() * rigidity2

	@staticmethod
	def backward(ctx, grad_h):
		global rigidity2
		src_V = ctx.saved_variables[0]
		src_E = ctx.saved_variables[1]

		lossD_gradient = pyDeform.DistanceFieldLoss_backward(src_V)
		lossR_gradient = pyDeform.GraphEdgeLoss_backward(src_V, src_E)

		return grad_h*(lossD_gradient + lossR_gradient * rigidity2), None


class GraphDeformLayer(nn.Module):
	def __init__(self, src_V, src_E, tar_V, tar_F, rigidity):
		super(GraphDeformLayer, self).__init__()
		
		pyDeform.InitializeDeformTemplate(tar_V, tar_F, 0, 64);

		pyDeform.NormalizeByTemplate(src_V)
		pyDeform.StoreGraphInformation(src_V, src_E)
		global rigidity2
		rigidity2 = rigidity * rigidity

	def forward(self, src_V, src_E):
		return GraphDeformFunction.apply(src_V, src_E)

def Finalize(src_V, src_F, src_E, src_to_graph, graph_V):
	pyDeform.NormalizeByTemplate(src_V)
	pyDeform.SolveLinear(src_V, src_F, src_E, src_to_graph, graph_V, rigidity2)
	pyDeform.DenormalizeByTemplate(src_V)