#ifndef MESH_TENSOR_H_
#define MESH_TENSOR_H_

#include <torch/extension.h>

#include <mesh.h>

void CopyMeshToTensor(const Mesh& m,
	torch::Tensor* ptensorV,
	torch::Tensor* ptensorF,
	int denormalize = 0);

void CopyTensorToMesh(const torch::Tensor& tensorV,
	const torch::Tensor& tensorF,
	Mesh* pm,
	int normalize = 0);

std::vector<torch::Tensor> LoadMesh(
	const char* filename);

void SaveMesh(const char* filename,
	const torch::Tensor& tensorV,
	const torch::Tensor& tensorF);

#endif