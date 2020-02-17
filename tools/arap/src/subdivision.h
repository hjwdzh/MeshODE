#ifndef ARAP_SUBDIVISION_H_
#define ARAP_SUBDIVISION_H_

#include "mesh.h"

#include <set>
#include <unordered_set>

class Subdivision
{
public:
	Subdivision();
	void Subdivide(const Mesh& mesh, double len_thres);
	void ComputeGeometryNeighbors(double len_thres);

	const Mesh* reference_mesh;
	Mesh subdivide_mesh;

	std::vector<int> parent_faces;
	std::set<std::pair<int, int> > geometry_neighbor_pairs;
	std::vector<int> vertex_component;
protected:
	void DelaunaySubdivision(
		std::unordered_set<int>* boundary_indices,
		std::vector<Vector3>& V,
		std::vector<Eigen::Vector3i>& F,
		Eigen::Vector3i& face,
		double len_thres, bool debug);
	void SubdivideFaces(std::vector<Vector3>& V,
		std::vector<Eigen::Vector3i>& F,
		std::vector<int>& parent_faces,
		std::vector<Eigen::Vector3i>& faces,
		int left_idx, int right_idx, double max_len, double len_thres);
	long long EdgeHash(int v1, int v2, int vsize = -1);
};

#endif