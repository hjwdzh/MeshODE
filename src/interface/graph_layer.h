#ifndef SHAPEDEFORM_INTERFACE_GRAPH_LAYER_H_
#define SHAPEDEFORM_INTERFACE_GRAPH_LAYER_H_

#include "deform_params.h"
#include "mesh_tensor.h"
#include "normalize.h"


void StoreGraphInformation(
	torch::Tensor tensorV,
	torch::Tensor tensorE,
	int param_id);

torch::Tensor GraphEdgeLoss_forward(
	torch::Tensor tensorV,
	torch::Tensor tensorE,
	int param_id);

torch::Tensor GraphEdgeLoss_backward(
	torch::Tensor tensorV,
	torch::Tensor tensorE,
	int param_id);

#endif