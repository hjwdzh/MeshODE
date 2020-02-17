#ifndef ARAP_MESHCOVER_H_
#define ARAP_MESHCOVER_H_

#include "mesh.h"

class MeshCover
{
public:
	MeshCover();	
	void Cover(const Mesh& watertight, Mesh& cad);
	void UpdateCover();
	const Mesh* watertight;
	Mesh cover;

	std::vector<int> findices;
	std::vector<Vector3> weights;

protected:
	long long EdgeHash(int v1, int v2, int vsize = -1);
};

#endif