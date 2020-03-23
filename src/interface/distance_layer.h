#ifndef SHAPEDEFORM_INTERFACE_DISTANCE_LAYER_H_
#define SHAPEDEFORM_INTERFACE_DISTANCE_LAYER_H_

#include "deform_params.h"
#include "mesh_tensor.h"
#include "normalize.h"

torch::Tensor DistanceFieldLoss_forward(
	torch::Tensor tensorV,
	int param_id);

torch::Tensor DistanceFieldLoss_backward(
	torch::Tensor tensorV,
	int param_id);

#endif