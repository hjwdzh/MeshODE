#ifndef SHAPEDEFORM_INTERFACE_LINEAR_LAYER_H_
#define SHAPEDEFORM_INTERFACE_LINEAR_LAYER_H_

#include <torch/extension.h>

void SolveLinear(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	torch::Tensor tensorE,
	torch::Tensor tensorRef,
	torch::Tensor tensorGraphV);

#endif