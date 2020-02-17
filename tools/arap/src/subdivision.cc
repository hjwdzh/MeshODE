#include "subdivision.h"

#include <fstream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

Subdivision::Subdivision()
{
	reference_mesh = 0;
}

void Subdivision::Subdivide(const Mesh& mesh, double len_thres)
{
	reference_mesh = &mesh;

	subdivide_mesh = mesh;

	auto& V = subdivide_mesh.V;
	auto& F = subdivide_mesh.F;

	// Build the edge to face graph
	std::unordered_map<long long, std::unordered_set<int> > edge_to_face;
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v1 = F[i][j];
			int v2 = F[i][(j + 1) % 3];
			long long edge_hash = EdgeHash(v1, v2);
			auto it = edge_to_face.find(edge_hash);
			if (it == edge_to_face.end()) {
				std::unordered_set<int> faces;
				faces.insert(i);
				edge_to_face[edge_hash] = faces;
			} else {
				it->second.insert(i);
			}
		}
	}

	std::vector<std::pair<int, int> > colors(F.size(), std::make_pair(-1, -1));
	int num_group = 0;
	for (int i = 0; i < F.size(); ++i) {
		if (colors[i].first == -1) {
			std::queue<int> q;
			q.push(i);
			colors[i] = std::make_pair(num_group, i);
			while (!q.empty()) {
				int fid = q.front();
				q.pop();
				for (int j = 0; j < 3; ++j) {
					int v1 = F[fid][j];
					int v2 = F[fid][(j + 1) % 3];
					long long edge_hash = EdgeHash(v1, v2);
					auto& neighbors = edge_to_face[edge_hash];
					for (auto& nf : neighbors) {
						if (colors[nf].first == -1) {
							colors[nf] = std::make_pair(num_group, nf);
							q.push(nf);
						}
					}
				}
			}
			num_group += 1;
		}
	}

	std::sort(colors.begin(), colors.end());
	std::vector<Eigen::Vector3i> faces_buffer(F.size());
	for (int i = 0; i < colors.size(); ++i) {
		faces_buffer[i] = F[colors[i].second];
	}

	F.clear();
	parent_faces.clear();

	connected_component_segments.clear();
	connected_component_segments.push_back(0);

	int left_idx = 0;
	while (left_idx < colors.size()) {
		double max_len = 0;
		int fid = colors[left_idx].first;
		int right_idx = left_idx;
		while (right_idx < colors.size() && colors[right_idx].first == fid) {
			for (int j = 0; j < 3; ++j) {
				int v1 = faces_buffer[right_idx][j];
				int v2 = faces_buffer[right_idx][(j + 1) % 3];
				auto diff = V[v1] - V[v2];
				max_len = std::max(max_len, diff.norm());
			}
			right_idx += 1;
		}

		SubdivideFaces(V, F, parent_faces, faces_buffer, left_idx, right_idx, max_len, len_thres);
		connected_component_segments.push_back(F.size());
		left_idx = right_idx;
	}

	vertex_component.resize(V.size());
	for (int i = 0; i < connected_component_segments.size() - 1; ++i) {
		int start_idx = connected_component_segments[i];
		int end_idx = connected_component_segments[i + 1];
		for (int j = start_idx; j < end_idx; ++j) {
			for (int k = 0; k < 3; ++k) {
				vertex_component[F[j][k]] = i;
			}
		}
	}
	ComputeGeometryNeighbors(len_thres / 8.0);
	ComputeGeometryNeighbors(len_thres / 4.0);
	ComputeGeometryNeighbors(len_thres / 2.0);
	ComputeGeometryNeighbors(len_thres / 1.0);
}

void Subdivision::ComputeGeometryNeighbors(double thres) {
	auto& vertices = subdivide_mesh.V;

	double step = thres;

	auto make_key = [&](const Vector3& v) {
		return std::make_pair(int(v[0]/step), std::make_pair(int(v[1]/step), int(v[2]/step)));
	};

	Vector3 diff[8] = {Vector3(0,0,0),
		Vector3(0,0,step),
		Vector3(0,step,0),
		Vector3(0,step,step),
		Vector3(step,0,0),
		Vector3(step,0,step),
		Vector3(step,step,0),
		Vector3(step,step,step)};

	std::map<std::pair<int, std::pair<int, int> >, std::unordered_map<int, int> > grids;
	for (int i = 0; i < vertices.size(); ++i) {
		for (int j = 0; j < 8; ++j) {
			auto v = vertices[i] + diff[j];
			auto key = make_key(v);
			auto it = grids.find(key);
			if (it == grids.end()) {
				std::unordered_map<int, int> m;
				m[vertex_component[i]] = i;
				grids[key] = m;
			} else {
				it->second[vertex_component[i]] = i;
			}
		}
	}

	std::vector<std::unordered_set<int> > links(vertices.size());
	for (auto& info : grids) {
		auto& l = info.second;
		for (auto it = l.begin(); it != l.end(); ++it) {
			auto next_it = it;
			next_it++;
			for (auto it1 = next_it; it1 != l.end(); ++it1) {
				int v1 = it->second;
				int v2 = it1->second;
				if (v1 > v2)
					std::swap(v1, v2);
				geometry_neighbor_pairs.insert(std::make_pair(v1, v2));
			}
		}
	}
}

void Subdivision::SubdivideFaces(std::vector<Vector3>& V,
	std::vector<Eigen::Vector3i>& F,
	std::vector<int>& parent_faces,
	std::vector<Eigen::Vector3i>& faces,
	int left_idx, int right_idx, double max_len, double len_thres) {

	int start_idx = F.size();
	for (int i = left_idx; i < right_idx; ++i) {
		F.push_back(faces[i]);
		parent_faces.push_back(i);
	}

	int vsize = 0;
	std::unordered_map<long long, int> edge_to_vid;
	auto split_edge = [&](int v1, int v2) {
		auto h = EdgeHash(v1, v2, vsize);
		auto it = edge_to_vid.find(h);
		if (it == edge_to_vid.end()) {
			Vector3 mid_v = (V[v1] + V[v2]) * FT(0.5);
			edge_to_vid[h] = V.size();
			V.push_back(mid_v);
			return (int)(V.size() - 1);
		}
		return it->second;
	};

	while (max_len > len_thres) {
		edge_to_vid.clear();
		int end_idx = F.size();
		vsize = V.size();
		for (int i = start_idx; i < end_idx; ++i) {
			int v0 = F[i][0];
			int v1 = F[i][1];
			int v2 = F[i][2];

			int nv0 = split_edge(v0, v1);
			int nv1 = split_edge(v1, v2);
			int nv2 = split_edge(v2, v0);

			F[i] = Eigen::Vector3i(nv0, nv1, nv2);
			F.push_back(Eigen::Vector3i(v0, nv0, nv2));
			F.push_back(Eigen::Vector3i(nv0, v1, nv1));
			F.push_back(Eigen::Vector3i(nv2, nv1, v2));

			parent_faces.push_back(parent_faces[i]);
			parent_faces.push_back(parent_faces[i]);
			parent_faces.push_back(parent_faces[i]);
		}
		max_len *= 0.5;
	}
}

long long Subdivision::EdgeHash(int v1, int v2, int vsize) {
	if (vsize == -1)
		vsize = subdivide_mesh.V.size();
	if (v1 < v2)
		return (long long)v1 * (long long)vsize + v2;
	else
		return (long long)v2 * (long long)vsize + v1;
}	
