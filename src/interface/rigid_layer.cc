#include "rigid_layer.h"

#include <vector>

#include <ATen/ATen.h>
#include <ceres/jet.h>
#include <torch/extension.h>

#include <uniformgrid.h>

void StoreRigidityInformation(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int param_id)
{
	auto& params = GetParams(param_id);
#ifndef USE_DOUBLE
	typedef float T;
	typedef Eigen::Vector3f V3;
#else
	typedef double T;
	typedef Eigen::Vector3d V3;
#endif
	const T* dataV = static_cast<const T*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());

	int f_size = tensorF.size(0);

	params.edge_offset.resize(f_size * 3);
	
	int offset = 0;
	for (int i = 0; i < f_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = dataF[i * 3 + j];
			int v1 = dataF[i * 3 + (j + 1) % 3];

			const T* v0_data = dataV + v0 * 3;
			const T* v1_data = dataV + v1 * 3;
			params.edge_offset[offset] = V3(
				v1_data[0] - v0_data[0],
				v1_data[1] - v0_data[1],
				v1_data[2] - v0_data[2]);
			offset += 1;
		}
	}
}

torch::Tensor RigidEdgeLoss_forward(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int param_id) {

	auto& params = GetParams(param_id);
#ifndef USE_DOUBLE
	typedef float T;
#else
	typedef double T;
#endif
	int f_size = tensorF.size(0);

	auto dataV = static_cast<const T*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());

#ifndef USE_DOUBLE
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
#else
	auto float_options = torch::TensorOptions().dtype(torch::kFloat64);
#endif
	torch::Tensor loss = torch::full({f_size * 3, 3}, 0, float_options);
	auto dataL = static_cast<T*>(loss.storage().data());

	for (int i = 0; i < f_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = dataF[(i * 3 + j)];
			int v1 = dataF[(i * 3 + (j + 1) % 3)];
			const T* v0_data = dataV + v0 * 3;
			const T* v1_data = dataV + v1 * 3;
			T* l = dataL + (i * 3 + j) * 3;
			l[0] = (v1_data[0] - v0_data[0] - params.edge_offset[i * 3 + j][0]);
			l[1] = (v1_data[1] - v0_data[1] - params.edge_offset[i * 3 + j][1]);
			l[2] = (v1_data[2] - v0_data[2] - params.edge_offset[i * 3 + j][2]);
			l[0] *= l[0];
			l[1] *= l[1];
			l[2] *= l[2];
		}
	}

	return loss;
}

torch::Tensor RigidEdgeLoss_backward(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int param_id) {

	auto& params = GetParams(param_id);
#ifndef USE_DOUBLE
	typedef float T;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
#else
	typedef double T;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat64);
#endif
	int v_size = tensorV.size(0);
	int f_size = tensorF.size(0);

	auto dataV = static_cast<const T*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());

	torch::Tensor loss = torch::full({v_size, 3}, /*value=*/0, float_options);
	auto dataL = static_cast<T*>(loss.storage().data());

	for (int i = 0; i < f_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = dataF[(i * 3 + j)];
			int v1 = dataF[(i * 3 + (j + 1) % 3)];
			const T* v0_data = dataV + v0 * 3;
			const T* v1_data = dataV + v1 * 3;

			T* l_v0 = dataL + v0 * 3;
			T* l_v1 = dataL + v1 * 3;

			for (int k = 0; k < 3; ++k) {
				l_v0[k] -= (v1_data[k] - v0_data[k]
					- params.edge_offset[i * 3 + j][k]);
				l_v1[k] += (v1_data[k] - v0_data[k]
					- params.edge_offset[i * 3 + j][k]);
			}
		}
	}

	return loss;	
}
