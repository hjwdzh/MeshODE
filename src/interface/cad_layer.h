#ifndef SHAPEDEFORM_INTERFACE_CAD_LAYER_H_
#define SHAPEDEFORM_INTERFACE_CAD_LAYER_H_

#include "deform_params.h"
#include "mesh_tensor.h"
#include "normalize.h"


void StoreCadInformation(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	torch::Tensor tensorE);

torch::Tensor CadEdgeLoss_forward(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	torch::Tensor tensorE);

torch::Tensor CadEdgeLoss_backward(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	torch::Tensor tensorE);

#endif