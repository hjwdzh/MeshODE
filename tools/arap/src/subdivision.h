#ifndef ARAP_SUBDIVISION_H_
#define ARAP_SUBDIVISION_H_

#include "mesh.h"

class Subdivision
{
public:
	Subdivision();
	void Subdivide(const Mesh& mesh, double len_thres);

	const Mesh* reference_mesh;
	Mesh subdivide_mesh;

protected:
	void SubdivideFaces(std::vector<Vector3>& V,
		std::vector<Eigen::Vector3i>& F,
		std::vector<Eigen::Vector3i>& faces,
		int left_idx, int right_idx, double max_len, double len_thres);
	long long EdgeHash(int v1, int v2, int vsize = -1);
};

#endif