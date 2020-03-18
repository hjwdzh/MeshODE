#include "normalize.h"

#include <torch/extension.h>

void NormalizeByTemplate(
	torch::Tensor tensorV)
{
#ifndef USE_DOUBLE
	typedef float T;
#else
	typedef double T;
#endif
	int v_size = tensorV.size(0);
	auto dataV = static_cast<T*>(tensorV.storage().data());
	auto& trans = params.trans;
	auto& scale = params.scale;
	for (int i = 0; i < v_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			dataV[i * 3 + j] = (dataV[i * 3 + j] - trans[j]) / scale;
		}
	}
}

void DenormalizeByTemplate(
	torch::Tensor tensorV)
{
#ifndef USE_DOUBLE
	typedef float T;
#else
	typedef double T;
#endif
	int v_size = tensorV.size(0);
	auto dataV = static_cast<T*>(tensorV.storage().data());
	auto& trans = params.trans;
	auto& scale = params.scale;
	for (int i = 0; i < v_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			dataV[i * 3 + j] = dataV[i * 3 + j] * scale + trans[j];
		}
	}
}
