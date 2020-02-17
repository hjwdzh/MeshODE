#ifndef ARAP_MESHCOVER_H_
#define ARAP_MESHCOVER_H_

#include "mesh.h"

class MeshCover
{
public:
	MeshCover();	
	void Cover(const Mesh& watertight, const Mesh& cad);

	Mesh cover;

protected:
	long long EdgeHash(int v1, int v2, int vsize = -1);
};

#endif