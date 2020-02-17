#ifndef ARAP_DEFORM_H_
#define ARAP_DEFORM_H_

#include "mesh.h"
#include "uniformgrid.h"

void Deform(Mesh& mesh, UniformGrid& grid, FT lambda = 1);
void DeformWithRot(Mesh& mesh, UniformGrid& grid, FT lambda = 1);

#endif