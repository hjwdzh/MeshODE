#include "distance_layer.h"

#include <ATen/ATen.h>
#include <ceres/jet.h>
#include <torch/extension.h>


torch::Tensor DistanceFieldLoss_forward(
	torch::Tensor tensorV) {

#ifndef USE_DOUBLE
	typedef float T;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
#else
	typedef double T;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat64);
#endif

	int v_size = tensorV.size(0);
	auto dataV = static_cast<const T*>(tensorV.storage().data());

	torch::Tensor loss = torch::full({v_size}, /*value=*/0, float_options);

	auto dataL = static_cast<T*>(loss.storage().data());
	for (int i = 0; i < v_size; ++i) {
#ifndef USE_DOUBLE
		dataL[i] = params.grid.DistanceFloat(dataV + i * 3);
#else
		dataL[i] = params.grid.distance(dataV + i * 3);
#endif
		dataL[i] *= dataL[i];
	}

	return loss;
}

torch::Tensor DistanceFieldLoss_backward(
	torch::Tensor tensorV) {

	int v_size = tensorV.size(0);
#ifndef USE_DOUBLE
	typedef float T;
	typedef Eigen::Vector3f V3;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
#else
	typedef double T;
	typedef Eigen::Vector3d V3;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat64);
#endif
	const T* dataV = static_cast<const T*>(tensorV.storage().data());

	torch::Tensor loss = torch::full({v_size, 3}, /*value=*/0, float_options);

	T* dataL = static_cast<T*>(loss.storage().data());
	for (int i = 0; i < v_size; ++i) {
		const T* v = dataV + i * 3;
		T* l = dataL + i * 3;

		ceres::Jet<T, 3> p[3] = {
			ceres::Jet<T, 3>(v[0], V3(1,0,0)),
			ceres::Jet<T, 3>(v[1], V3(0,1,0)),
			ceres::Jet<T, 3>(v[2], V3(0,0,1))
		};

#ifndef USE_DOUBLE
		auto vd = params.grid.DistanceFloat(p);
#else
		auto vd = params.grid.distance(p);
#endif
		vd *= vd;

		l[0] = vd.v[0] * 0.5;
		l[1] = vd.v[1] * 0.5;
		l[2] = vd.v[2] * 0.5;
	}

	return loss;
}