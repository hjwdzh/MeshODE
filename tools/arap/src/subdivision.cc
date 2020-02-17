#include "subdivision.h"

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

	printf("Number of groups %d %d %d\n", num_group, F.size(), V.size());
	std::sort(colors.begin(), colors.end());
	std::vector<Eigen::Vector3i> faces_buffer(F.size());
	for (int i = 0; i < colors.size(); ++i) {
		faces_buffer[i] = F[colors[i].second];
	}

	F.clear();
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

		SubdivideFaces(V, F, faces_buffer, left_idx, right_idx, max_len, len_thres);
		left_idx = right_idx;
	}
	printf("Number of groups %d %d %d\n", num_group, F.size(), V.size());
}

void Subdivision::SubdivideFaces(std::vector<Vector3>& V,
	std::vector<Eigen::Vector3i>& F,
	std::vector<Eigen::Vector3i>& faces,
	int left_idx, int right_idx, double max_len, double len_thres) {

	int start_idx = F.size();
	for (int i = left_idx; i < right_idx; ++i) {
		F.push_back(faces[i]);
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
