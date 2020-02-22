#include "meshcover.h"

#include <unordered_map>

#include <igl/point_mesh_squared_distance.h>

MeshCover::MeshCover()
{
}

void MeshCover::Cover(Mesh& watertight, Subdivision& sub) {
	auto& cad = sub.subdivide_mesh;
	MatrixX V1(cad.V.size(), 3), V2(watertight.V.size(), 3);
	Eigen::MatrixXi F2(watertight.F.size(), 3);

	for (int i = 0; i < cad.V.size(); ++i)
		V1.row(i) = cad.V[i];

	for (int i = 0; i < watertight.V.size(); ++i)
		V2.row(i) = watertight.V[i];

	for (int i = 0; i < watertight.F.size(); ++i)
		F2.row(i) = watertight.F[i];

	MatrixX sqrD;
	Eigen::VectorXi I;
	MatrixX C;

	auto t1 = std::chrono::high_resolution_clock::now();
	igl::point_mesh_squared_distance(V1,V2,F2,sqrD,I,C);
	auto t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    std::cout << duration << "\n";

	exit(0);
	findices.clear();
	weights.clear();
	for (int i = 0; i < I.size(); ++i) {
		Eigen::MatrixXd weight;
		int find = I[i];
		igl::barycentric_coordinates(C.row(i), V2.row(F2(find, 0)),
			V2.row(F2(find, 1)), V2.row(F2(find, 2)), weight);
		findices.push_back(find);
		weights.push_back(weight.row(0));
	}
}

void MeshCover::UpdateCover(Mesh& watertight, Subdivision& sub) {
	// Linear Estimation
	auto& V = sub.subdivide_mesh.V;
	auto& F = sub.subdivide_mesh.F;

	std::unordered_map<long long, FT> trips;
	int num_entries = V.size();
	MatrixX B = MatrixX::Zero(num_entries, 3);
	auto add_entry_A = [&](int x, int y, FT w) {
		long long key = (long long)x * (long long)num_entries + (long long)y;
		auto it = trips.find(key);
		if (it == trips.end())
			trips[key] = w;
		else
			it->second += w;
	};
	auto add_entry_B = [&](int m, const Vector3& v) {
		B.row(m) += v;
	};

	for (int i = 0; i < findices.size(); ++i) {
		auto& f = watertight.F[findices[i]];
		const Vector3& v0 = watertight.V[f[0]];
		const Vector3& v1 = watertight.V[f[1]];
		const Vector3& v2 = watertight.V[f[2]];

		Vector3 v = v0 * weights[i][0] + v1 * weights[i][1] + v2 * weights[i][2];

		add_entry_A(i, i, 1);
		add_entry_B(i, v);
	}

	FT regular = 0;
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = F[i][j];
			int v1 = F[i][(j + 1) % 3];

			double reg = regular;
			add_entry_A(v0, v0, reg);
			add_entry_A(v0, v1, -reg);
			add_entry_A(v1, v0, -reg);
			add_entry_A(v1, v1, reg);
			add_entry_B(v0, reg * (V[v0] - V[v1]));
			add_entry_B(v1, reg * (V[v1] - V[v0]));
		}
	}

	for (auto& p : sub.geometry_neighbor_pairs) {
		int v0 = p.first;
		int v1 = p.second;

		add_entry_A(v0, v0, regular);
		add_entry_A(v0, v1, -regular);
		add_entry_A(v1, v0, -regular);
		add_entry_A(v1, v1, regular);
		add_entry_B(v0, regular * (V[v0] - V[v1]));
		add_entry_B(v1, regular * (V[v1] - V[v0]));
	}

	Eigen::SparseMatrix<FT> A(num_entries, num_entries);
	std::vector<T> tripletList;
	for (auto& m : trips) {
		tripletList.push_back(T(m.first / num_entries, m.first % num_entries, m.second));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());

	Eigen::SparseLU<Eigen::SparseMatrix<FT>> solver;
    solver.analyzePattern(A);

    solver.factorize(A);

    //std::vector<Vector3> NV(V.size());
    for (int j = 0; j < 3; ++j) {
        VectorX result = solver.solve(B.col(j));

        for (int i = 0; i < result.rows(); ++i) {
        	V[i][j] = result[i];
        }
    }
}
