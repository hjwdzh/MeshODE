#ifndef ARAP_DEFORMER_H_
#define ARAP_DEFORMER_H_

#include <memory>

#include "callback.h"
#include "mesh.h"
#include "subdivision.h"
#include "uniformgrid.h"

class Deformer {
public:
	Deformer();

	Deformer(FT lambda = (FT)1.0, CallBackFunc func = 0);

	void Deform(const UniformGrid& grid, Mesh* mesh);

	void DeformWithRot(const UniformGrid& grid, Mesh* mesh);

	void DeformSubdivision(const UniformGrid& grid, Subdivision* sub);

	void ReverseDeform(const Mesh& tar, Mesh* src);

private:
	FT lambda_;
	std::shared_ptr<TerminateWhenSuccessCallback> callback_;
};
#endif