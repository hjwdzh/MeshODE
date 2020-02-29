#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>

#include "deformer.h"
#include "mesh.h"
#include "uniformgrid.h"

namespace py = pybind11;

std::vector<torch::Tensor> LoadMesh(
	const char* filename) {
	Mesh src;
	src.ReadOBJ(filename);
	auto int_options = torch::TensorOptions().dtype(torch::kInt32);
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);

	auto& V = src.GetV();
	auto& F = src.GetF();

	torch::Tensor tensorV = torch::full({(long long)V.size(), 3}, /*value=*/0, float_options);
	torch::Tensor tensorF = torch::full({(long long)F.size(), 3}, /*value=*/0, int_options);

	auto dataV = static_cast<float*>(tensorV.storage().data());
	auto dataF = static_cast<int*>(tensorF.storage().data());

	for (int i = 0; i < V.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			dataV[i * 3 + j] = V[i][j];
		}
	}

	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			dataF[i * 3 + j] = F[i][j];
		}
	}

	return {tensorV, tensorF};
}

PYBIND11_MODULE(pyDeform, m) {
	m.def("LoadMesh", &LoadMesh);
}
