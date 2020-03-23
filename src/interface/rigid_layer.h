#ifndef SHAPEDEFORM_INTERFACE_RIGID_LAYER_H_
#define SHAPEDEFORM_INTERFACE_RIGID_LAYER_H_

#include "deform_params.h"
#include "mesh_tensor.h"
#include "normalize.h"


void StoreRigidityInformation(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int param_id);

torch::Tensor RigidEdgeLoss_forward(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int param_id);

torch::Tensor RigidEdgeLoss_backward(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int param_id);

#endif