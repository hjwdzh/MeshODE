#include "mesh.h"

#include <fstream>
#include <strstream>

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
		os << "f " << F[i][0] + 1 << " " << F[i][1] + 1 << " " << F[i][2] + 1 << "\n";
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
	scale = std::max(max_p[0] - min_p[0], std::max(max_p[1] - min_p[1], max_p[2] - min_p[2])) * 1.1;
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
	MatrixX P(grid.N * grid.N * grid.N, 3);
	int offset = 0;
	for (int i = 0; i < grid.N; ++i) {
		for (int j = 0; j < grid.N; ++j) {
			for (int k = 0; k < grid.N; ++k) {
				P.row(offset) = Vector3(FT(k) / grid.N, FT(j) / grid.N, FT(i) / grid.N);
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
	for (int i = 0; i < grid.N; ++i) {
		for (int j = 0; j < grid.N; ++j) {
			for (int k = 0; k < grid.N; ++k) {
				grid.distances[i][j][k] = sqrt(sqrD[offset]);
				offset += 1;
			}
		}
	}

}

void Mesh::FromDistanceField(UniformGrid& grid) {
	VectorX S(grid.N * grid.N * grid.N);
	MatrixX GV(grid.N * grid.N * grid.N, 3);
	int offset = 0;
	for (int i = 0; i < grid.N; ++i) {
		for (int j = 0; j < grid.N; ++j) {
			for (int k = 0; k < grid.N; ++k) {
				S[offset] = grid.distances[i][j][k];
				GV.row(offset) = Vector3(k, j, i);
				offset += 1;
			}
		}
	}
	MatrixX SV;
	Eigen::MatrixXi SF;
	igl::copyleft::marching_cubes(S,GV,grid.N,grid.N,grid.N,SV,SF);
	V.resize(SV.rows());
	F.resize(SF.rows());
	for (int i = 0; i < SV.rows(); ++i)
		V[i] = SV.row(i) / (FT)grid.N;
	for (int i = 0; i < SF.rows(); ++i)
		F[i] = SF.row(i);
}

void Mesh::MergeDuplex() {
	std::map<std::pair<int, std::pair<int, int> >, int> grid_map;
	auto key = [&](const Vector3& v) {
		return std::make_pair(v[0] * 1e6, std::make_pair(v[1] * 1e6, v[2] * 1e6));
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
}

void Mesh::ReflectionSymmetrize() {
	int face_num = F.size();
	int vert_num = V.size();
	for (int i = 0; i < vert_num; ++i) {
		V.push_back(Vector3(-V[i][0], V[i][1], V[i][2]));
	}
	for (int i = 0; i < face_num; ++i) {
		F.push_back(Eigen::Vector3i(F[i][0] + vert_num, F[i][1] + vert_num, F[i][2] + vert_num));
	}
}