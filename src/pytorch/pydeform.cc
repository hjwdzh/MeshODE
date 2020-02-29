#include <ATen/ATen.h>

#include <ceres/jet.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

#include <vector>

#include "mesh.h"
#include "uniformgrid.h"

namespace py = pybind11;

struct DeformParams
{
	DeformParams()
	: lambda(1.0), scale(1.0), trans(0, 0, 0)
	{}
	Mesh ref;
	UniformGrid grid;
	FT lambda;

	FT scale;
	Vector3 trans;

	std::vector<Eigen::Vector3f> edge_offset;
};

DeformParams params;

void CopyMeshToTensor(const Mesh& m,
	torch::Tensor* ptensorV,
	torch::Tensor* ptensorF,
	int denormalize = 0)
{
	auto& V = m.GetV();
	auto& F = m.GetF();
	auto& tensorV = *ptensorV;
	auto& tensorF = *ptensorF;

	auto int_options = torch::TensorOptions().dtype(torch::kInt32);
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
	tensorV = torch::full({(long long)V.size(), 3}, /*value=*/0, float_options);
	tensorF = torch::full({(long long)F.size(), 3}, /*value=*/0, int_options);

	auto dataV = static_cast<float*>(tensorV.storage().data());
	auto dataF = static_cast<int*>(tensorF.storage().data());

	auto trans = m.GetTranslation();
	auto scale = m.GetScale();
	for (int i = 0; i < V.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			if (denormalize)
				dataV[i * 3 + j] = V[i][j] * scale + trans[j];
			else
				dataV[i * 3 + j] = V[i][j];
		}
	}

	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			dataF[i * 3 + j] = F[i][j];
		}
	}
}

void CopyTensorToMesh(const torch::Tensor& tensorV,
	const torch::Tensor& tensorF,
	Mesh* pm,
	int normalize = 0)
{
	auto& m = *pm;

	auto& V = m.GetV();
	auto& F = m.GetF();

	auto dataV = static_cast<const float*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());

	int v_size = tensorV.size(0);
	int f_size = tensorF.size(0);
	V.resize(v_size);
	F.resize(f_size);

	for (int i = 0; i < V.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			V[i][j] = dataV[i * 3 + j];
		}
	}

	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			F[i][j] = dataF[i * 3 + j];
		}
	}

	if (normalize) {
		m.Normalize();
	}
}

std::vector<torch::Tensor> LoadMesh(
	const char* filename) {
	Mesh src;
	src.ReadOBJ(filename);

	torch::Tensor tensorV;
	torch::Tensor tensorF;

	CopyMeshToTensor(src, &tensorV, &tensorF, 0);

	return {tensorV, tensorF};
}

void InitializeDeformTemplate(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int symmetry,
	int grid_resolution,
	FT lambda) {

	params.ref = Mesh();


	if (symmetry)
		params.ref.ReflectionSymmetrize();

	params.lambda = lambda;
	params.grid = UniformGrid(grid_resolution);

	CopyTensorToMesh(tensorV, tensorF, &params.ref, 1);
	params.ref.ConstructDistanceField(params.grid);

	params.scale = params.ref.GetScale();
	params.trans = params.ref.GetTranslation();
}

void NormalizeByTemplate(
	torch::Tensor tensorV)
{
	int v_size = tensorV.size(0);
	auto dataV = static_cast<float*>(tensorV.storage().data());
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
	int v_size = tensorV.size(0);
	auto dataV = static_cast<float*>(tensorV.storage().data());
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
	const float* dataV = static_cast<const float*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());

	int f_size = tensorF.size(0);

	params.edge_offset.resize(f_size * 3);
	int offset = 0;
	for (int i = 0; i < f_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = dataF[i * 3 + j];
			int v1 = dataF[i * 3 + (j + 1) % 3];
			const float* v0_data = dataV + v0 * 3;
			const float* v1_data = dataV + v1 * 3;
			params.edge_offset[offset] = Eigen::Vector3f(
				v1_data[0] - v0_data[0],
				v1_data[1] - v0_data[1],
				v1_data[2] - v0_data[2]);
			offset += 1;
		}
	}
}

torch::Tensor DistanceFieldLoss_forward(
	torch::Tensor tensorV) {

	int v_size = tensorV.size(0);
	auto dataV = static_cast<const float*>(tensorV.storage().data());

	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor loss = torch::full({v_size}, /*value=*/0, float_options);

	auto dataL = static_cast<float*>(loss.storage().data());
	for (int i = 0; i < v_size; ++i) {
		dataL[i] = params.grid.distance(dataV + i * 3);
		dataL[i] *= dataL[i];
	}

	return loss;
}

torch::Tensor DistanceFieldLoss_backward(
	torch::Tensor tensorV) {

	int v_size = tensorV.size(0);
	const float* dataV = static_cast<const float*>(tensorV.storage().data());

	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor loss = torch::full({v_size, 3}, /*value=*/0, float_options);

	float* dataL = static_cast<float*>(loss.storage().data());
	for (int i = 0; i < v_size; ++i) {
		const float* v = dataV + i * 3;
		float* l = dataL + i * 3;

		ceres::Jet<float, 3> p[3] = {
			ceres::Jet<float, 3>(v[0], Eigen::Vector3f(1,0,0)),
			ceres::Jet<float, 3>(v[1], Eigen::Vector3f(0,1,0)),
			ceres::Jet<float, 3>(v[2], Eigen::Vector3f(0,0,1))
		};

		auto vd = params.grid.distance(p);
		vd *= vd;
		l[0] = vd.v[0];
		l[1] = vd.v[1];
		l[2] = vd.v[2];
	}

	return loss;
}

PYBIND11_MODULE(pyDeform, m) {
	m.def("LoadMesh", &LoadMesh);
	m.def("InitializeDeformTemplate", &InitializeDeformTemplate);
	m.def("NormalizeByTemplate", &NormalizeByTemplate);
	m.def("DenormalizeByTemplate", &DenormalizeByTemplate);
	m.def("DistanceFieldLoss_forward", &DistanceFieldLoss_forward);
	m.def("DistanceFieldLoss_backward", &DistanceFieldLoss_backward);
	m.def("StoreRigidityInformation", &StoreRigidityInformation);
}

