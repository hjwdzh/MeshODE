#include "rigid_layer.h"

#include <vector>

#include <ATen/ATen.h>
#include <ceres/jet.h>
#include <torch/extension.h>

#include <uniformgrid.h>

void StoreCadInformation(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	torch::Tensor tensorE,
	int param_id)
{
#ifndef USE_DOUBLE
	typedef float T;
	typedef Eigen::Vector3f V3;
#else
	typedef double T;
	typedef Eigen::Vector3d V3;
#endif
	const T* dataV = static_cast<const T*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());
	auto dataE = static_cast<const int*>(tensorE.storage().data());

	int e_size = tensorE.size(0);
	int f_size = tensorF.size(0);

	auto& params = GetParams(param_id);

	params.edge_offset.resize(e_size + f_size * 3);
	params.edge_lambda.resize(e_size + f_size * 3);
	
	int offset = 0;
	for (int i = 0; i < e_size; ++i) {
		int v0 = dataE[i * 2];
		int v1 = dataE[i * 2 + 1];

		const T* v0_data = dataV + v0 * 3;
		const T* v1_data = dataV + v1 * 3;
		params.edge_offset[offset] = V3(
			v1_data[0] - v0_data[0],
			v1_data[1] - v0_data[1],
			v1_data[2] - v0_data[2]);

		params.edge_lambda[offset] = (2e-2
			/ (params.edge_offset[offset].norm()+1e-8));

		offset += 1;

	}
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

			params.edge_offset[offset] = V3(
				v1_data[0] - v0_data[0],
				v1_data[1] - v0_data[1],
				v1_data[2] - v0_data[2]);

			params.edge_lambda[offset] = (2e-2
				/ (params.edge_offset[offset].norm()+1e-8));

			offset += 1;
		}
	}
}

torch::Tensor CadEdgeLoss_forward(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	torch::Tensor tensorE,
	int param_id) {

	auto& params = GetParams(param_id);
#ifndef USE_DOUBLE
	typedef float T;
#else
	typedef double T;
#endif
	int f_size = tensorF.size(0);
	int e_size = tensorE.size(0);

	auto dataV = static_cast<const T*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());
	auto dataE = static_cast<const int*>(tensorE.storage().data());

#ifndef USE_DOUBLE
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
#else
	auto float_options = torch::TensorOptions().dtype(torch::kFloat64);
#endif
	torch::Tensor loss = torch::full({e_size + f_size * 3, 3}, 0,
		float_options);
	auto dataL = static_cast<T*>(loss.storage().data());

	int offset = 0;

	for (int i = 0; i < e_size; ++i) {
		int v0 = dataE[i * 2];
		int v1 = dataE[i * 2 + 1];
		const T* v0_data = dataV + v0 * 3;
		const T* v1_data = dataV + v1 * 3;
		T* l = dataL + offset * 3;
		l[0] = (v1_data[0] - v0_data[0] - params.edge_offset[offset][0])
			 * params.edge_lambda[offset];
		l[1] = (v1_data[1] - v0_data[1] - params.edge_offset[offset][1])
			 * params.edge_lambda[offset];
		l[2] = (v1_data[2] - v0_data[2] - params.edge_offset[offset][2])
			 * params.edge_lambda[offset];
		l[0] *= l[0];
		l[1] *= l[1];
		l[2] *= l[2];

		offset += 1;
	}

	for (int i = 0; i < f_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = dataF[(i * 3 + j)];
			int v1 = dataF[(i * 3 + (j + 1) % 3)];
			const T* v0_data = dataV + v0 * 3;
			const T* v1_data = dataV + v1 * 3;
			T* l = dataL + offset * 3;
			l[0] = (v1_data[0] - v0_data[0] - params.edge_offset[offset][0])
				 * params.edge_lambda[offset];
			l[1] = (v1_data[1] - v0_data[1] - params.edge_offset[offset][1])
				 * params.edge_lambda[offset];
			l[2] = (v1_data[2] - v0_data[2] - params.edge_offset[offset][2])
				 * params.edge_lambda[offset];
			l[0] *= l[0];
			l[1] *= l[1];
			l[2] *= l[2];

			offset += 1;
		}
	}

	return loss;
}

torch::Tensor CadEdgeLoss_backward(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	torch::Tensor tensorE,
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
	int e_size = tensorE.size(0);

	auto dataV = static_cast<const T*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());
	auto dataE = static_cast<const int*>(tensorE.storage().data());

	torch::Tensor loss = torch::full({v_size, 3}, /*value=*/0, float_options);
	auto dataL = static_cast<T*>(loss.storage().data());

	int offset = 0;
	for (int i = 0; i < e_size; ++i) {
		int v0 = dataE[i * 2];
		int v1 = dataE[i * 2 + 1];

		const T* v0_data = dataV + v0 * 3;
		const T* v1_data = dataV + v1 * 3;

		T* l_v0 = dataL + v0 * 3;
		T* l_v1 = dataL + v1 * 3;

		T lambda = params.edge_lambda[offset];
		lambda *= lambda;
		for (int k = 0; k < 3; ++k) {
			l_v0[k] -= (v1_data[k] - v0_data[k]
				- params.edge_offset[offset][k]) * lambda;
			l_v1[k] += (v1_data[k] - v0_data[k]
				- params.edge_offset[offset][k]) * lambda;
		}

		offset += 1;
	}
	for (int i = 0; i < f_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = dataF[(i * 3 + j)];
			int v1 = dataF[(i * 3 + (j + 1) % 3)];
			const T* v0_data = dataV + v0 * 3;
			const T* v1_data = dataV + v1 * 3;

			T* l_v0 = dataL + v0 * 3;
			T* l_v1 = dataL + v1 * 3;

			T lambda = params.edge_lambda[offset];
			lambda *= lambda;
			for (int k = 0; k < 3; ++k) {
				l_v0[k] -= (v1_data[k] - v0_data[k]
					- params.edge_offset[offset][k]) * lambda;
				l_v1[k] += (v1_data[k] - v0_data[k]
					- params.edge_offset[offset][k]) * lambda;
			}

			offset += 1;
		}
	}

	return loss;	
}
