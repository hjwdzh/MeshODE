#ifndef DEFORM_PARAMS_H_
#define DEFORM_PARAMS_H_

#include <mesh.h>

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
#else
	std::vector<Eigen::Vector3d> edge_offset;
#endif
};

#endif