#ifndef ARAP_MESH_H_
#define ARAP_MESH_H_

#include "types.h"

#include "uniformgrid.h"

class Mesh
{
public:
	Mesh();

	// FILE IO
	void ReadOBJ(const char* filename);
	void WriteOBJ(const char* filename, bool normalized = false);

	void ReflectionSymmetrize();
	// Normalize and Denormalize
	void Normalize();
	void ApplyTransform(Mesh& m);

	// Conversion between Distance Field
	void ConstructDistanceField(UniformGrid& grid);
	void FromDistanceField(UniformGrid& grid);

	// Merge
	void MergeDuplex();

	// Main Deformation function
	void Deform(UniformGrid& grid);

	std::vector<Vector3> V;
	std::vector<Eigen::Vector3i> F;

	FT scale;
	Vector3 pos;
};

#endif