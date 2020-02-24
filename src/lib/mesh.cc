#include "mesh.h"

#include <fstream>
#include <strstream>
#include <set>

#include <igl/copyleft/marching_cubes.h>
#include <igl/point_mesh_squared_distance.h>

Mesh::Mesh()
	: scale(1.0), pos(0, 0, 0)
{}

void Mesh::ReadOBJ(const char* filename) {
	std::ifstream is(filename);
	char buffer[256];
	while (is.getline(buffer, 256)) {
		std::strstream str;
		str << buffer;
		str >> buffer;
		if (strcmp(buffer, "v") == 0) {
			FT x, y, z;
			str >> x >> y >> z;
			V.push_back(Vector3(x, y, z));
		}
		else if (strcmp(buffer, "f") == 0) {
			Eigen::Vector3i f;
			for (int j = 0; j < 3; ++j) {
				str >> buffer;
				int id = 0;
				int p = 0;
				int l = strlen(buffer);
				while (p < l && buffer[p] != '/') {
					id = id * 10 + (buffer[p] - '0');
					p += 1;
				}
				f[j] = id - 1;
			}
			F.push_back(f);
		}
	}
}

void Mesh::WriteOBJ(const char* filename, bool normalized) {
	std::ofstream os(filename);
	for (int i = 0; i < V.size(); ++i) {
		Vector3 v;
		if (!normalized)
			v = V[i] * scale + pos;
		else
			v = V[i];
		os << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
	}
	for (int i = 0; i < F.size(); ++i) {
		os << "f " << F[i][0] + 1 << " "
		   << F[i][1] + 1 << " "
		   << F[i][2] + 1 << "\n";
	}
	os.close();
}

void Mesh::Normalize() {
	FT min_p[3], max_p[3];
	for (int j = 0; j < 3; ++j) {
		min_p[j] = 1e30;
		max_p[j] = -1e30;
		for (int i = 0; i < V.size(); ++i) {
			if (V[i][j] < min_p[j])
				min_p[j] = V[i][j];
			if (V[i][j] > max_p[j])
				max_p[j] = V[i][j];
		}
	}
	scale = std::max(max_p[0] - min_p[0],
		std::max(max_p[1] - min_p[1], max_p[2] - min_p[2])) * 1.1;
	for (int j = 0; j < 3; ++j)
		pos[j] = min_p[j] - 0.05 * scale;
	for (auto& v : V) {
		v = (v - pos) / scale;
	}
	for (int j = 0; j < 3; ++j) {
		min_p[j] = 1e30;
		max_p[j] = -1e30;
		for (int i = 0; i < V.size(); ++i) {
			if (V[i][j] < min_p[j])
				min_p[j] = V[i][j];
			if (V[i][j] > max_p[j])
				max_p[j] = V[i][j];
		}
	}
}

void Mesh::ApplyTransform(Mesh& m) {
	pos = m.pos;
	scale = m.scale;
	for (auto& v : V) {
		v = (v - pos) / scale;
	}
}
void Mesh::ConstructDistanceField(UniformGrid& grid) {
	int grid_n = grid.Dimension();
	MatrixX P(grid_n * grid_n * grid_n, 3);
	int offset = 0;
	for (int i = 0; i < grid_n; ++i) {
		for (int j = 0; j < grid_n; ++j) {
			for (int k = 0; k < grid_n; ++k) {
				P.row(offset) = Vector3(FT(k) / grid_n,
					FT(j) / grid_n, FT(i) / grid_n);
				offset += 1;
			}
		}
	}

	MatrixX V2(V.size(), 3);
	for (int i = 0; i < V.size(); ++i)
		V2.row(i) = V[i];

	Eigen::MatrixXi F2(F.size(), 3);
	for (int i = 0; i < F.size(); ++i)
		F2.row(i) = F[i];

	MatrixX N(F.size(), 3);
	for (int i = 0; i < F.size(); ++i) {
		Vector3 x = V[F[i][1]] - V[F[i][0]];
		Vector3 y = V[F[i][2]] - V[F[i][0]];
		N.row(i) = x.cross(y).normalized();
	}

	VectorX sqrD;
	Eigen::VectorXi I;
	MatrixX C;
	igl::point_mesh_squared_distance(P,V2,F2,sqrD,I,C);

	offset = 0;
	for (int i = 0; i < grid_n; ++i) {
		for (int j = 0; j < grid_n; ++j) {
			for (int k = 0; k < grid_n; ++k) {
				grid.SetDistance(i, j, k, sqrt(sqrD[offset]));
				offset += 1;
			}
		}
	}

}

void Mesh::FromDistanceField(UniformGrid& grid) {
	int grid_n = grid.Dimension();
	VectorX S(grid_n * grid_n * grid_n);
	MatrixX GV(grid_n * grid_n * grid_n, 3);
	int offset = 0;
	for (int i = 0; i < grid_n; ++i) {
		for (int j = 0; j < grid_n; ++j) {
			for (int k = 0; k < grid_n; ++k) {
				S[offset] = grid.GetDistance(i, j, k);
				GV.row(offset) = Vector3(k, j, i);
				offset += 1;
			}
		}
	}
	MatrixX SV;
	Eigen::MatrixXi SF;
	igl::copyleft::marching_cubes(S,GV,grid_n,grid_n,grid_n,SV,SF);
	V.resize(SV.rows());
	F.resize(SF.rows());
	for (int i = 0; i < SV.rows(); ++i)
		V[i] = SV.row(i) / (FT)grid_n;
	for (int i = 0; i < SF.rows(); ++i)
		F[i] = SF.row(i);
}

void Mesh::MergeDuplex() {
	std::map<std::pair<int, std::pair<int, int> >, int> grid_map;
	auto key = [&](const Vector3& v) {
		return std::make_pair(v[0] * 1e6,
			std::make_pair(v[1] * 1e6, v[2] * 1e6));
	};
	int top = 0;
	std::vector<int> shrink(V.size());
	for (int i = 0; i < V.size(); ++i) {
		auto k = key(V[i]);
		auto it = grid_map.find(k);
		if (it == grid_map.end()) {
			shrink[i] = top;
			grid_map[k] = top;
			V[top] = V[i];
			top++;
		} else {
			shrink[i] = it->second;
		}
	}
	V.resize(top);
	
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			F[i][j] = shrink[F[i][j]];
		}
	}

	std::set<std::pair<int, std::pair<int, int> > > face_set;
	top = 0;
	for (int i = 0; i < F.size(); ++i) {
		int v0 = F[i][0];
		int v1 = F[i][1];
		int v2 = F[i][2];
		if (v0 == v1 || v1 == v2 || v2 == v0)
			continue;
		if (v0 > v1)
			std::swap(v0, v1);
		if (v0 > v2)
			std::swap(v0, v2);
		if (v1 > v2)
			std::swap(v1, v2);
		auto k = std::make_pair(v0, std::make_pair(v1, v2));
		if (face_set.count(k))
			continue;
		face_set.insert(k);
		F[top++] = F[i];
	}
	F.resize(top);
}

void Mesh::ReflectionSymmetrize() {
	int face_num = F.size();
	int vert_num = V.size();
	for (int i = 0; i < vert_num; ++i) {
		V.push_back(Vector3(-V[i][0], V[i][1], V[i][2]));
	}
	for (int i = 0; i < face_num; ++i) {
		F.push_back(Eigen::Vector3i(F[i][0] + vert_num,
			F[i][1] + vert_num, F[i][2] + vert_num));
	}
}

void Mesh::ComputeFaceNormals() {
	NF.resize(F.size());
	for (auto& n : NF)
		n = Eigen::Vector3d(0, 0, 0);
	for (int i = 0; i < F.size(); ++i) {
		Eigen::Vector3d& v0 = V[F[i][0]];
		Eigen::Vector3d& v1 = V[F[i][1]];
		Eigen::Vector3d& v2 = V[F[i][2]];
		NF[i] = (v1 - v0).cross(v2 - v0);
		NF[i] /= NF[i].norm();
	}
}

void Mesh::ComputeVertexNormals() {
	NV.resize(V.size());
	for (auto& n : NV)
		n = Eigen::Vector3d(0, 0, 0);
	for (int i = 0; i < F.size(); ++i) {
		Eigen::Vector3d& v0 = V[F[i][0]];
		Eigen::Vector3d& v1 = V[F[i][1]];
		Eigen::Vector3d& v2 = V[F[i][2]];
		Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
		NV[F[i][0]] += n;
		NV[F[i][1]] += n;
		NV[F[i][2]] += n;
	}
	for (int i = 0; i < NV.size(); ++i) {
		NV[i] /= NV[i].norm();
	}
}

void Mesh::LogStatistics(const char* filename) {
	std::set<std::pair<int, int> > edges;
	int duplicate = 0;
	int boundary = 0;
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = F[i][j];
			int v1 = F[i][(j+1)%3];
			auto key = std::make_pair(v0, v1);
			if (edges.count(key)) {
				duplicate += 1;
			}
			edges.insert(key);
		}
	}

	std::ofstream os(filename);
	int top = 1;
	for (auto& info : edges) {
		auto rinfo = std::make_pair(info.second, info.first);
		if (!edges.count(rinfo)) {
			boundary += 1;
			auto v0 = V[info.first];
			auto v1 = V[info.second];
			os << "v " << v0[0] << " " << v0[1] << " " << v0[2] << "\n";
			os << "v " << v1[0] << " " << v1[1] << " " << v1[2] << "\n";
			os << "l " << top << " " << top + 1 << "\n";
			top += 2;
		}
	}
	os.close();
}

void Mesh::RemoveDegenerated() {
	int top = 0;
	for (int i = 0; i < F.size(); ++i) {
		Eigen::Vector3d v0 = V[F[i][0]];
		Eigen::Vector3d v1 = V[F[i][1]];
		Eigen::Vector3d v2 = V[F[i][2]];
		Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
		if (n.norm() > 0) {
			F[top++] = F[i];
		}
	}
	F.resize(top);
}