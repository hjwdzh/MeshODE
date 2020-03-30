#include "subdivision.h"

#include <iostream>
#include <fstream>
#include <queue>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

#include "delaunay.h"
#include "linear.h"

Subdivision::Subdivision()
{
}

void Subdivision::ApplyTransform(const Mesh& mesh) {
	subdivide_mesh_.ApplyTransform(mesh);
	auto pos = mesh.GetTranslation();
	auto scale = mesh.GetScale();
	for (auto& v : representative_vertices_) {
		v = (v - pos) / scale;
	}
}

void Subdivision::Subdivide(const Mesh& mesh, double len_thres)
{
	subdivide_mesh_ = mesh;

	auto& subdivide_mesh = subdivide_mesh_;
	auto& V = subdivide_mesh.GetV();
	auto& F = subdivide_mesh.GetF();

	std::unordered_set<int> boundary_vertices;

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

	int v_num = V.size();
	int vsize = 0x7fffffff;

	std::unordered_map<long long, std::vector<int> > edge_subdivision_indices;
	for (int i = 0; i < faces_buffer.size(); ++i) {
		long long hash[3];
		for (int j = 0; j < 3; ++j) {
			int v0 = faces_buffer[i][j];
			int v1 = faces_buffer[i][(j + 1) % 3];
			auto h = EdgeHash(v0, v1, vsize);
			hash[j] = h;
			if (edge_subdivision_indices.count(h) == 0) {
				Vector3 diff = V[v1] - V[v0];
				int num_splits = diff.norm() / len_thres + 1;
				diff /= (FT)num_splits;
				std::vector<int> vindices;
				vindices.push_back(v0);
				for (int j = 1; j < num_splits; ++j) {
					boundary_vertices.insert(V.size());
					vindices.push_back(V.size());
					V.push_back(V[v0] + diff * j);
				}
				vindices.push_back(v1);
				edge_subdivision_indices[h] = vindices;
			}
		}
		std::vector<int> boundary_indices[3];
		for (int j = 0; j < 3; ++j) {
			boundary_indices[j] = edge_subdivision_indices[hash[j]];
		}
		DelaunaySubdivision(boundary_indices, V, F, faces_buffer[i], len_thres);
	}

	internal_vertices_.resize(V.size(), 0);
	for (int i = v_num; i < V.size(); ++i) {
		if (!boundary_vertices.count(i))
			internal_vertices_[i] = 1;
	}
}

void Subdivision::DelaunaySubdivision(
	std::vector<int>* boundary_indices,
	std::vector<Vector3>& V,
	std::vector<Eigen::Vector3i>& F,
	Eigen::Vector3i& face,
	double len_thres) {

	Eigen::Vector3d v0 = V[face[0]];
	Eigen::Vector3d v1 = V[face[1]];
	Eigen::Vector3d v2 = V[face[2]];

	Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
	n /= n.norm();
	std::unordered_set<int> merged_indices;
	std::unordered_map<int, Eigen::Vector3d> curved_point;
	for (int i = 0; i < 3; ++i) {
		Eigen::Vector3d x = V[face[i]];
		Eigen::Vector3d y = V[face[(i+1)%3]];
		Eigen::Vector3d dir = n.cross(y - x);
		dir /= dir.norm();
		Eigen::Vector3d c = (y + x) * 0.5 + 1e3 * dir;
		double len = (c - x).norm();
		for (auto& p : boundary_indices[i]) {
			merged_indices.insert(p);
			Eigen::Vector3d diff = V[p] - c;
			diff = diff / diff.norm() * len + c;
			curved_point[p] = diff;
		}
	}

	std::vector<int> vindices;
	std::vector<Eigen::Vector3d> points;
	auto c = (v0 + v1 + v2) / 3.0;

	double max_len = std::max((c - v0).norm(),
		std::max((c - v1).norm(), (c - v2).norm()));

	int min_axis = 0;
	for (int i = 1; i < 3; ++i) {
		if (std::abs(n[i]) < std::abs(n[min_axis])) {
			min_axis = 1;
		}
	}
	Vector3 tx(0, 0, 0);
	tx[min_axis] = 1;
	tx = tx.cross(n);
	tx /= tx.norm();
	Vector3 ty = n.cross(tx);

	for (auto p : merged_indices) {
		auto v = V[p];
		vindices.push_back(p);
		Eigen::Vector3d diff = v - c;
		diff = diff / diff.norm() * max_len + c;
		//points.push_back(v);
		points.push_back(curved_point[p]);
	}

	double v0x = 0;
	double v0y = 0;
	double v1x = (v1-v0).dot(tx) / len_thres;
	double v1y = (v1-v0).dot(ty) / len_thres;
	double v2x = (v2-v0).dot(tx) / len_thres;
	double v2y = (v2-v0).dot(ty) / len_thres;

	int minX = (std::min(v0x, std::min(v1x, v2x)));
	int minY = (std::min(v0y, std::min(v1y, v2y)));
	int maxX = (std::max(v0x, std::max(v1x, v2x))) + 0.999999f;
	int maxY = (std::max(v0y, std::max(v1y, v2y))) + 0.999999f;

	int count = 0;
	for (int py = minY; py <= maxY; ++py) {
		for (int px = minX; px <= maxX; ++px) {
			double w1, w2, w3;
			calculateBarycentricCoordinate(v0x, v0y, v1x, v1y, v2x, v2y,
				px, py, w1, w2, w3);

			if (w1 > 0.0 && w1 < 1.0 &&
				w2 > 0.0 && w2 < 1.0 &&
				w3 > 0.0 && w3 < 1.0) {
				Vector3 rand_p = v0
							   + tx * (px * len_thres)
							   + ty * (py * len_thres);
				vindices.push_back(V.size());
				V.push_back(rand_p);
				points.push_back(rand_p);
				count += 1;
			}
		}
	}


	Eigen::MatrixXd V2D(vindices.size(), 2);
	for (int i = 0; i < vindices.size(); ++i) {
		V2D(i, 0) = (points[i] - v0).dot(tx);
		V2D(i, 1) = (points[i] - v0).dot(ty);
	}

	Eigen::MatrixXi F2D;

	Delaunay2D(V2D, F2D);
	for (int i = 0; i < F2D.rows(); ++i) {
		int v[3];
		v[0] = vindices[F2D(i, 0)];
		v[1] = vindices[F2D(i, 1)];
		v[2] = vindices[F2D(i, 2)];
		bool boundary_triangle = false;

		for (int j = 0; j < 3; ++j) {
			for (int k = j + 1; k < 3; ++k) {
				if ((V[v[j]] - V[v[k]]).norm() > 3 * len_thres) {
					boundary_triangle = true;
				}
			}
		}
		if (!boundary_triangle) {
			F.push_back(Eigen::Vector3i(v[0], v[1], v[2]));
		}
	}
}

void Subdivision::ComputeGeometryNeighbors(double thres) {
	auto& subdivide_mesh = subdivide_mesh_;
	auto& vertices = subdivide_mesh.GetV();
	auto& faces = subdivide_mesh.GetF();

	double step = thres;

	auto make_key = [&](const Vector3& v) {
		return std::make_pair(int(v[0]/step),
			std::make_pair(int(v[1]/step), int(v[2]/step)));
	};

	Vector3 diff[8] = {Vector3(0,0,0),
		Vector3(0,0,step),
		Vector3(0,step,0),
		Vector3(0,step,step),
		Vector3(step,0,0),
		Vector3(step,0,step),
		Vector3(step,step,0),
		Vector3(step,step,step)};

	std::map<std::pair<int, std::pair<int, int> >,
		std::unordered_set<int> > grids;

	for (int i = 0; i < vertices.size(); ++i) {
		for (int j = 0; j < 8; ++j) {
			auto v = vertices[i] + diff[j];
			auto key = make_key(v);
			auto it = grids.find(key);
			if (it == grids.end()) {
				std::unordered_set<int> m;
				m.insert(i);
				grids[key] = m;
			} else {
				it->second.insert(i);
			}
		}
	}

	std::vector<std::unordered_set<int> > links(vertices.size());
	int count = 0;
	for (auto& info : grids) {
		auto& l = info.second;
		if (l.size() < 1)
			continue;
		if (l.size() == 2) {
			auto it = l.begin();
			it++;
			int v1 = *l.begin(), v2 = *it;
			if (v1 > v2)
				std::swap(v1, v2);
			if (v1 == v2)
				continue;
			count += 1;
			geometry_neighbor_pairs_.insert(std::make_pair(v1, v2));
			continue;
		}
		Eigen::MatrixXd gridV(l.size(), 3);
		int top = 0;
		std::vector<int> vindices(l.size());
		for (auto p : l) {
			vindices[top] = p;
			gridV.row(top++) = vertices[p];
		}

		std::vector<std::pair<int, int> > gridE;
		Delaunay3D(gridV, gridE);
		count += gridE.size();
		for (auto& e : gridE) {
			int v1 = vindices[e.first];
			int v2 = vindices[e.second];
			if (v1 > v2)
				std::swap(v1, v2);
			if (v1 == v2)
				continue;
			geometry_neighbor_pairs_.insert(std::make_pair(v1, v2));
		}
	}
	for (int i = 0; i < faces.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v1 = faces[i][j];
			int v2 = faces[i][(j + 1) % 3];
			if (v1 > v2)
				std::swap(v1, v2);
			if (v1 == v2)
				continue;
			geometry_neighbor_pairs_.insert(std::make_pair(v1, v2));
		}
	}
}

void Subdivision::ComputeRepresentativeGraph(double thres) {
	double step = thres;

	auto make_key = [&](const Vector3& v) {
		return std::make_pair(int(v[0]/step),
			std::make_pair(int(v[1]/step), int(v[2]/step)));
	};
	std::map<std::pair<int, std::pair<int, int> >,
		std::unordered_set<int> > grids;

	auto& V = subdivide_mesh_.GetV();
	auto& F = subdivide_mesh_.GetF();
	for (int i = 0; i < V.size(); ++i) {
		auto k = make_key(V[i]);
		auto it = grids.find(k);
		if (it != grids.end()) {
			it->second.insert(i);
		} else {
			std::unordered_set<int> u;
			u.insert(i);
			grids[k] = u;
		}
	}

	representative_vertices_.reserve(grids.size());
	representative_reference_.resize(V.size());
	for (auto& info : grids) {
		Vector3 p(0, 0, 0);
		for (auto& id : info.second) {
			p += V[id];
			representative_reference_[id] = representative_vertices_.size();
		}
		p /= (double)info.second.size();
		representative_vertices_.push_back(p);
	}
	for (auto& info : geometry_neighbor_pairs_) {
		int v0 = representative_reference_[info.first];
		int v1 = representative_reference_[info.second];
		if (v0 == v1)
			continue;
		if (v0 > v1)
			std::swap(v0, v1);
		representative_edges_.insert(std::make_pair(v0, v1));
	}
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = representative_reference_[F[i][j]];
			int v1 = representative_reference_[F[i][(j + 1) % 3]];
			if (v0 == v1)
				continue;
			if (v0 > v1)
				std::swap(v0, v1);
			representative_edges_.insert(std::make_pair(v0, v1));
		}
	}

}

long long Subdivision::EdgeHash(int v1, int v2, int vsize) {
	if (vsize == -1)
		vsize = subdivide_mesh_.GetV().size();
	if (v1 < v2)
		return (long long)v1 * (long long)vsize + v2;
	else
		return (long long)v2 * (long long)vsize + v1;
}	


void Subdivision::SmoothInternal() {
	auto& subdivide_mesh = subdivide_mesh_;
	auto& V = subdivide_mesh.GetV();
	auto& F = subdivide_mesh.GetF();
	std::vector<std::unordered_set<int> > links(V.size());
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = F[i][j];
			int v1 = F[i][(j + 1) % 3];
			links[v1].insert(v0);
			links[v0].insert(v1);
		}
	}
	for (int iter = 0; iter < 3; ++iter) {
		auto V_buf = V;
		for (int i = 0; i < V.size(); ++i) {
			if (!internal_vertices_[i]) {
				continue;
			}
			if (links[i].size() == 0)
				continue;
			Eigen::Vector3d v(0, 0, 0);
			for (auto& l : links[i]) {
				v += V_buf[l];
			}
			v /= links[i].size();
			V[i] = v;
		}
	}
}

double Subdivision::calculateSignedArea2(double ax, double ay,
	double bx, double by,
	double cx, double cy) {
	return ((cx - ax) * (by - ay) - (bx - ax) * (cy - ay));
}

void Subdivision::calculateBarycentricCoordinate(
	double ax, double ay,
	double bx, double by,
	double cx, double cy,
	double px, double py,
	double& alpha, double& beta, double& gamma) {
	double beta_tri = calculateSignedArea2(ax, ay, px, py, cx, cy);
	double gamma_tri = calculateSignedArea2(ax, ay, bx, by, px, py);
	double tri_inv = 1.0f / calculateSignedArea2(ax, ay, bx, by, cx, cy);
	beta = beta_tri * tri_inv;
	gamma = gamma_tri * tri_inv;
	alpha = 1.0 - beta - gamma;
}

void Subdivision::LinearSolve() {
	auto& V = subdivide_mesh_.GetV();
	auto& F = subdivide_mesh_.GetF();

	LinearEstimation(V, F, geometry_neighbor_pairs_.begin(),
		geometry_neighbor_pairs_.end(),
		representative_reference_,
		representative_vertices_);
}