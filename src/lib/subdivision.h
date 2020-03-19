#ifndef SHAPEDEFORM_SUBDIVISION_H_
#define SHAPEDEFORM_SUBDIVISION_H_

#include "mesh.h"

#include <set>
#include <unordered_set>

class Subdivision
{
public:
	Subdivision();
	void Subdivide(const Mesh& mesh, double len_thres);
	void ApplyTransform(const Mesh& mesh);
	void ComputeGeometryNeighbors(double len_thres);
	void ComputeRepresentativeGraph(double len_thres);

	void LinearSolve();
	void SmoothInternal();
	Mesh& GetMesh() { return subdivide_mesh_; }
	const std::set<std::pair<int, int> >& Neighbors() const {
		return geometry_neighbor_pairs_;
	}
	std::vector<Vector3>& GraphV() {
		return representative_vertices_;
	}
	std::set<std::pair<int, int> >& GraphE() {
		return representative_edges_;
	}

protected:
	void DelaunaySubdivision(
		std::vector<int>* boundary_indices,
		std::vector<Vector3>& V,
		std::vector<Eigen::Vector3i>& F,
		Eigen::Vector3i& face,
		double len_thres);

	long long EdgeHash(int v1, int v2, int vsize = -1);

	double calculateSignedArea2(double ax, double ay,
		double bx, double by,
		double cx, double cy);

	void calculateBarycentricCoordinate(
		double ax, double ay,
		double bx, double by,
		double cx, double cy,
		double px, double py,
		double& alpha, double& beta, double& gamma);

private:
	Mesh subdivide_mesh_;

	std::set<std::pair<int, int> > geometry_neighbor_pairs_;
	std::vector<int> internal_vertices_;

	std::vector<int> representative_reference_;
	std::vector<Vector3> representative_diffs_;
	std::vector<Vector3> representative_vertices_;
	std::set<std::pair<int, int> > representative_edges_;


};

#endif