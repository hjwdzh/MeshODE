#include "deform_params.h"

#include "mesh_tensor.h"

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