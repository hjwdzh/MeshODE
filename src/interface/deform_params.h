#ifndef SHAPEDEFORM_INTERFACE_DEFORM_PARAMS_H_
#define SHAPEDEFORM_INTERFACE_DEFORM_PARAMS_H_

#include <mesh.h>
#include <torch/extension.h>

struct DeformParams
{
	DeformParams()
	: scale(1.0), trans(0, 0, 0)
	{}
	Mesh ref;
	UniformGrid grid;

	FT scale;
	Vector3 trans;

#ifndef USE_DOUBLE
	std::vector<Eigen::Vector3f> edge_offset;
	std::vector<float> edge_lambda;
#else
	std::vector<Eigen::Vector3d> edge_offset;
	std::vector<double> edge_lambda;
#endif
};

int CreateParams();
DeformParams& GetParams(int param_id);

int InitializeDeformTemplate(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int symmetry,
	int grid_resolution);

#endif