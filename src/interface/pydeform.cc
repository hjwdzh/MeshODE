#include <ATen/ATen.h>

#include <ceres/jet.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

#include <vector>

#include "deform_params.h"
#include "mesh_tensor.h"
#include <uniformgrid.h>

namespace py = pybind11;
//#define USE_DOUBLE

DeformParams params;

void InitializeDeformTemplate(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int symmetry,
	int grid_resolution) {

	params.ref = Mesh();


	if (symmetry)
		params.ref.ReflectionSymmetrize();

	params.grid = UniformGrid(grid_resolution);

	CopyTensorToMesh(tensorV, tensorF, &params.ref, 1);
	params.ref.ConstructDistanceField(params.grid);

	params.scale = params.ref.GetScale();
	params.trans = params.ref.GetTranslation();

}

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

void StoreRigidityInformation(
	torch::Tensor tensorV,
	torch::Tensor tensorF)
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

	int v_size = tensorV.size(0);
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

torch::Tensor EdgeLoss_forward(
	torch::Tensor tensorV,
	torch::Tensor tensorF) {

#ifndef USE_DOUBLE
	typedef float T;
#else
	typedef double T;
#endif
	int v_size = tensorV.size(0);
	int f_size = tensorF.size(0);

	auto dataV = static_cast<const T*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());

#ifndef USE_DOUBLE
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
#else
	auto float_options = torch::TensorOptions().dtype(torch::kFloat64);
#endif
	torch::Tensor loss = torch::full({f_size, 3, 3}, 0, float_options);
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

torch::Tensor EdgeLoss_backward(
	torch::Tensor tensorV,
	torch::Tensor tensorF) {

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

PYBIND11_MODULE(pyDeform, m) {
	m.def("LoadMesh", &LoadMesh);
	m.def("SaveMesh", &SaveMesh);
	m.def("InitializeDeformTemplate", &InitializeDeformTemplate);
	m.def("NormalizeByTemplate", &NormalizeByTemplate);
	m.def("DenormalizeByTemplate", &DenormalizeByTemplate);
	m.def("DistanceFieldLoss_forward", &DistanceFieldLoss_forward);
	m.def("DistanceFieldLoss_backward", &DistanceFieldLoss_backward);
	m.def("EdgeLoss_forward", &EdgeLoss_forward);
	m.def("EdgeLoss_backward", &EdgeLoss_backward);
	m.def("StoreRigidityInformation", &StoreRigidityInformation);
}

