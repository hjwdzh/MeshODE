#ifndef ARAP_MESHCOVER_H_
#define ARAP_MESHCOVER_H_

#include "subdivision.h"

class MeshCover
{
public:
	MeshCover();	
	void Cover(Mesh& watertight, Subdivision& sub);
	void UpdateCover(Mesh& watertight, Subdivision& sub);
	
	std::vector<int> findices;
	std::vector<Vector3> weights;

protected:
	long long EdgeHash(int v1, int v2, int vsize = -1);
};

#endif