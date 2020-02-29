#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>

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

	printf("%d %d\n", v_size, f_size);
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

PYBIND11_MODULE(pyDeform, m) {
	m.def("LoadMesh", &LoadMesh);
	m.def("InitializeDeformTemplate", &InitializeDeformTemplate);
	m.def("NormalizeByTemplate", &NormalizeByTemplate);
}

