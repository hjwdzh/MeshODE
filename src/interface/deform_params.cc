#include "deform_params.h"

#include <vector>

#include "mesh_tensor.h"

std::vector<DeformParams> g_params;
int CreateParams() {
	g_params.push_back(DeformParams());
	return (int)g_params.size() - 1;
}
DeformParams& GetParams(int param_id) {
	return g_params[param_id];
}

int InitializeDeformTemplate(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int symmetry,
	int grid_resolution) {

	int param_id = CreateParams();
	auto& params = GetParams(param_id);

	params.ref = Mesh();


	if (symmetry)
		params.ref.ReflectionSymmetrize();

	params.grid = UniformGrid(grid_resolution);

	CopyTensorToMesh(tensorV, tensorF, &params.ref, 1);
	params.ref.ConstructDistanceField(params.grid);

	params.scale = params.ref.GetScale();
	params.trans = params.ref.GetTranslation();

	return param_id;
}