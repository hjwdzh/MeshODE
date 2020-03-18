#ifndef SHAPEDEFORM_INTERFACE_NORMALIZE_H_
#define SHAPEDEFORM_INTERFACE_NORMALIZE_H_

#include "deform_params.h"

void NormalizeByTemplate(
	torch::Tensor tensorV);

void DenormalizeByTemplate(
	torch::Tensor tensorV);

#endif