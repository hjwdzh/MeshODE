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
	void RemoveDegenerated();
	void MergeDuplex();

	// Main Deformation function
	void Deform(UniformGrid& grid);

	void ComputeVertexNormals();
	void ComputeFaceNormals();

	void LogStatistics(const char* filename);

	std::vector<Vector3>& GetV() { return V_; }
	const std::vector<Vector3>& GetV() const { return V_; }

	std::vector<Eigen::Vector3i>& GetF() { return F_; }
	const std::vector<Eigen::Vector3i>& GetF() const { return F_; }

	std::vector<Eigen::Vector3d>& GetNF() { return NF_; }
	const std::vector<Eigen::Vector3d>& GetNF() const { return NF_; }

	std::vector<Eigen::Vector3d>& GetNV() { return NV_; }	
	const std::vector<Eigen::Vector3d>& GetNV() const { return NV_; }	

	Vector3 GetTranslation() const { return pos_; };
	FT GetScale() const { return scale_; };

private:
	std::vector<Vector3> V_;
	std::vector<Eigen::Vector3i> F_;

	std::vector<Eigen::Vector3d> NF_, NV_;
	FT scale_;
	Vector3 pos_;
};

#endif