import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform

device = None

class GraphLossFunction(Function):
	@staticmethod
	def forward(ctx, src_V, src_E, rigidity2, param_id):
		global device
		pid = param_id.tolist()

		test_V = torch.from_numpy(src_V.data.cpu().numpy())
		lossD = pyDeform.DistanceFieldLoss_forward(test_V, int(pid)) * 0.5
		lossR = pyDeform.GraphEdgeLoss_forward(test_V, src_E, int(pid)) * 0.5
		mask_D = lossD < (0.5 * 0.03 * 0.03)

		variables = [src_V, src_E, rigidity2, param_id, mask_D]
		ctx.save_for_backward(*variables)

		return (lossD.sum() + lossR.sum() * rigidity2.tolist()).to(device)

	@staticmethod
	def backward(ctx, grad_h):
		global device
		src_V = ctx.saved_variables[0]
		src_E = ctx.saved_variables[1]
		rigidity2 = ctx.saved_variables[2]
		param_id = ctx.saved_variables[3].tolist()
		mask_D = ctx.saved_variables[4]
		mask_D = mask_D.view(mask_D.shape[0],1)
		mask_D = torch.cat((mask_D,mask_D,mask_D),axis=1)

		test_V = torch.from_numpy(src_V.data.cpu().numpy())
		lossD_gradient = pyDeform.DistanceFieldLoss_backward(test_V, param_id)
		lossR_gradient = pyDeform.GraphEdgeLoss_backward(test_V, src_E, param_id)

		lossD_gradient *= mask_D

		return (grad_h*(lossD_gradient + lossR_gradient*rigidity2.tolist())).to(device),\
			None, None, None


class GraphLossLayer(nn.Module):
	def __init__(self, src_V, src_E, tar_V, tar_F, rigidity, d=torch.device('cpu')):
		super(GraphLossLayer, self).__init__()

		global device
		device = d
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
	pyDeform.SolveLinear(src_V, src_F, src_E, src_to_graph, graph_V, rigidity, 0)
	pyDeform.DenormalizeByTemplate(src_V, param_id.tolist())