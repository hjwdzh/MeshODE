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

	void Deform(Mesh& mesh, UniformGrid& grid);

	void DeformWithRot(Mesh& mesh, UniformGrid& grid);

	void DeformSubdivision(Subdivision& sub, UniformGrid& grid);

	void ReverseDeform(Mesh& src, Mesh& tar);

private:
	FT lambda_;
	std::shared_ptr<TerminateWhenSuccessCallback> callback_;
};
#endif