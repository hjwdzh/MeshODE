#include <iostream>
#include <fstream>
#include <unordered_map>

#include <igl/point_mesh_squared_distance.h>

#include "mesh.h"
#include "meshcover.h"
#include "uniformgrid.h"

// flags
int GRID_RESOLUTION = 64;
int MESH_RESOLUTION = 5000;

// main function
int main(int argc, char** argv) {	
	if (argc < 5) {
		printf("./coverage_deform source.obj reference.obj output.obj "
			"[GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] "
			"[lambda=1] [symmetry=0]\n");
		return 0;
	}

	Mesh src, ref, cad;
	src.ReadOBJ(argv[1]);
	ref.ReadOBJ(argv[2]);

	int symmetry = 0;
	if (argc > 7) {
		sscanf(argv[6], "%d", &symmetry);
	}

	if (symmetry)
		ref.ReflectionSymmetrize();

	if (argc > 4)
		sscanf(argv[4], "%d", &GRID_RESOLUTION);

	if (argc > 5)
		sscanf(argv[5], "%d", &MESH_RESOLUTION);

	FT lambda = 1;
	if (argc > 6)
		sscanf(argv[6], "%lf", &lambda);

	//Get number of vertices and faces
	std::cout << "Source:\t\t" << "Num vertices: " << src.GetV().size()
		<< "\tNum faces: " << src.GetF().size() << std::endl;
	std::cout<<"Reference:\t" << "Num vertices: " <<ref.GetV().size()
		<< "\tNum faces: " << ref.GetF().size() <<std::endl <<std::endl;


	Mesh src_copy = src;
	Mesh ref_copy = ref;
	UniformGrid grid(GRID_RESOLUTION);
	src_copy.Normalize();
	ref_copy.ApplyTransform(src_copy);
	ref.ApplyTransform(src_copy);
	src_copy.ConstructDistanceField(grid);

	MatrixX sqrD1, sqrD2;
	Eigen::VectorXi I1, I2;
	MatrixX C1, C2;

	auto& src_copy_V = src_copy.GetV();
	auto& src_copy_F = src_copy.GetF();
	auto& ref_copy_V = ref_copy.GetV();
	auto& ref_copy_F = ref_copy.GetF();
	auto& ref_V = ref.GetV();

	Eigen::MatrixXd V1(ref_copy_V.size(), 3);
	for (int i = 0; i < ref_copy_V.size(); ++i) {
		V1.row(i) = ref_copy_V[i];
	}

	Eigen::MatrixXi F1(ref_copy_F.size(), 3);
	for (int i = 0; i < ref_copy_F.size(); ++i) {
		F1.row(i) = ref_copy_F[i];
	}

	Eigen::MatrixXd V2(src_copy_V.size(), 3);
	for (int i = 0; i < src_copy_V.size(); ++i) {
		V2.row(i) = src_copy_V[i];
	}

	Eigen::MatrixXi F2(src_copy_F.size(), 3);
	for (int i = 0; i < src_copy_F.size(); ++i) {
		F2.row(i) = src_copy_F[i];
	}

	igl::point_mesh_squared_distance(V1,V2,F2,sqrD1,I1,C1);
	igl::point_mesh_squared_distance(V2,V1,F1,sqrD2,I2,C2);

	
	std::unordered_map<long long, FT> trips;
	
	int num_entries = src_copy_V.size();
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

	src_copy.ComputeFaceNormals();
	ref_copy.ComputeVertexNormals();

	//src_copy.WriteOBJ("../scan2cad/c1/point-src.obj", true);
	//ref_copy.WriteOBJ("../scan2cad/c1/point-ref.obj", true);
	//std::ofstream os("../scan2cad/c1/point.obj");
	int count = 0;
	for (int i = 0; i < C1.rows(); ++i) {
		auto v_ref = ref_copy_V[i];
		if (sqrt(sqrD1(i, 0)) < 0.1) {
			int find = I1[i];
			bool consistent = true;
			for (int j = 0; j < 3; ++j) {
				int vid = src_copy_F[find][j];
				Eigen::Vector3d v_s = C2.row(vid);
				if ((v_s - v_ref).norm() > 5e-2)
					consistent = false;
			}
			if (consistent) {
				MatrixX weight;
				igl::barycentric_coordinates(C1.row(i),
					V2.row(F2(find, 0)),
					V2.row(F2(find, 1)),
					V2.row(F2(find, 2)), weight);

				for (int j = 0; j < 3; ++j) {
					Eigen::Vector3d p0(0,0,0), p1(0,0,0);
					for (int k = 0; k < 3; ++k) {
						p0 += weight(0, k) * src_copy_V[F2(find, k)];
					}
					p1 = ref_V[i];
					Eigen::Vector3d diff = (p0 - p1);
					diff /= diff.norm();
					if (std::abs(diff.dot(src_copy.GetNF()[find])) < 0.5) {
						continue;
					}
					if (std::abs(diff.dot(ref_copy.GetNV()[i])) < 0.5) {
						continue;
					}
					
					for (int k = 0; k < 3; ++k) {
						add_entry_A(F2(find, j), F2(find, k),
							weight(0, j) * weight(0, k));
					}
					add_entry_B(F2(find, j), ref_V[i] * weight(0, j));

					count += 1;
				}
			}
		}
	}
	
	for (int i = 0; i < src_copy_V.size(); ++i) {
		add_entry_A(i, i, 1e-6);
		add_entry_B(i, src_copy_V[i] * 1e-6);
	}

	FT regular = lambda;
	for (int i = 0; i < src_copy_F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = src_copy_F[i][j];
			int v1 = src_copy_F[i][(j + 1) % 3];

			double reg = regular;
			add_entry_A(v0, v0, reg);
			add_entry_A(v0, v1, -reg);
			add_entry_A(v1, v0, -reg);
			add_entry_A(v1, v1, reg);
			add_entry_B(v0, reg * (src_copy_V[v0] - src_copy_V[v1]));
			add_entry_B(v1, reg * (src_copy_V[v1] - src_copy_V[v0]));
		}
	}

	SpMat A(num_entries, num_entries);
	std::vector<T> tripletList;
	for (auto& m : trips) {
		tripletList.push_back(T(m.first / num_entries,
			m.first % num_entries, m.second));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());

	Eigen::SparseLU<Eigen::SparseMatrix<FT>> solver;
    solver.analyzePattern(A);

    solver.factorize(A);

    //std::vector<Vector3> NV(V.size());
    for (int j = 0; j < 3; ++j) {
        VectorX result = solver.solve(B.col(j));

        for (int i = 0; i < result.rows(); ++i) {
        	src_copy_V[i][j] = result[i];
        }
    }
    src_copy.WriteOBJ(argv[3]);

	return 0;
}
