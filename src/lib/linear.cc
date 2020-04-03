#include "linear.h"

#include <set>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

template<class Iter>
void LinearEstimation(std::vector<Vector3>& V,
	const std::vector<Eigen::Vector3i>& F,
	const Iter& E_begin, const Iter& E_end,
	const std::vector<int>& references,
	const std::vector<Vector3>& graphV,
	double rigidity)
{
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

	std::vector<std::vector<int> > grid_cell(graphV.size());
	for (int i = 0; i < references.size(); ++i)
		grid_cell[references[i]].push_back(i);


	for (int i = 0; i < grid_cell.size(); ++i) {
		double weight = 1.0 / grid_cell[i].size();
		double weight2 = weight * weight;
		for (int j = 0; j < grid_cell[i].size(); ++j) {
			for (int k = 0; k < grid_cell[i].size(); ++k) {
				add_entry_A(grid_cell[i][j], grid_cell[i][k], weight2);
			}
		}
		for (int j = 0; j < grid_cell[i].size(); ++j) {
			add_entry_B(grid_cell[i][j], weight * graphV[i]);
		}
	}

	for (int i = 0; i < V.size(); ++i) {
		double weight = 1e-6;
		add_entry_A(i, i, weight);
		add_entry_B(i, weight *
			graphV[references[i]]);
	}


	FT regular = rigidity;

	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = F[i][j];
			int v1 = F[i][(j + 1) % 3];
			double reg = regular * 2e-2 / ((V[v0] - V[v1]).norm() + 1e-8);
			reg *= reg;
			add_entry_A(v0, v0, reg);
			add_entry_A(v0, v1, -reg);
			add_entry_A(v1, v0, -reg);
			add_entry_A(v1, v1, reg);
			add_entry_B(v0, reg * (V[v0] - V[v1]));
			add_entry_B(v1, reg * (V[v1] - V[v0]));
		}
	}
	for (auto it = E_begin; it != E_end; ++it) {
		auto& info = *it;
		int v0 = info.first;
		int v1 = info.second;

		double reg = regular * 2e-2 / ((V[v0] - V[v1]).norm() + 1e-8);
		reg *= reg;
		add_entry_A(v0, v0, reg);
		add_entry_A(v0, v1, -reg);
		add_entry_A(v1, v0, -reg);
		add_entry_A(v1, v1, reg);
		add_entry_B(v0, reg * (V[v0] - V[v1]));
		add_entry_B(v1, reg * (V[v1] - V[v0]));
	}

	SpMat A(num_entries, num_entries);
	std::vector<T> tripletList;
	for (auto& m : trips) {
		tripletList.push_back(T(m.first / num_entries,
			m.first % num_entries, m.second));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<FT>> solver;
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

void LinearEstimationWithRot(double* V, int* F, double* TV,
	int num_V, int num_F) {
	std::vector<int> E(num_F * 6);
	int num_E = 0;
	for (int i = 0; i < num_F; ++i) {
		for (int j = 0; j < 3; ++j) {
			E[num_E++] = F[i * 3 + j];
			E[num_E++] = F[i * 3 + (j + 1) % 3];
		}
	}
	num_E /= 2;
	std::vector<std::set<int> > links(num_V);
	std::vector<Eigen::Matrix3d> rotations(num_V);
	std::vector<double> scales(num_V);

	for (int i = 0; i < num_E; ++i) {
		int v1 = E[i * 2];
		int v2 = E[i * 2 + 1];
		links[v1].insert(v2);
		links[v2].insert(v1);
	}

	for (int i = 0; i < num_V; ++i) {
		Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

		double* current_v = V + i * 3;
		double* current_tv = TV + i * 3;

		double len_origin = 0, len_current = 0;
		for (auto& p : links[i]) {
			double* next_v = V + p * 3;
			double* next_tv = TV + p * 3;

			Eigen::Vector3d d1(next_v[0] - current_v[0],
				next_v[1] - current_v[1],
				next_v[2] - current_v[2]);
			Eigen::Vector3d d2(next_tv[0] - current_tv[0],
				next_tv[1] - current_tv[1],
				next_tv[2] - current_tv[2]);

			len_origin += d1.norm();
			len_current += d2.norm();
			covariance += d2 * d1.transpose();
		}
		double scale = len_current / (len_origin + 1e-8);
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance,
			Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d VT = svd.matrixV().transpose();
		Eigen::Matrix3d R = U * VT;
		rotations[i] = R;
		scales[i] = scale;
	}

	std::unordered_map<long long, FT> trips;
	
	int num_entries = num_V;
	Eigen::MatrixXd B = Eigen::MatrixXd::Zero(num_entries, 3);
	auto add_entry_A = [&](int x, int y, FT w) {
		long long key = (long long)x * (long long)num_entries + (long long)y;
		auto it = trips.find(key);
		if (it == trips.end())
			trips[key] = w;
		else
			it->second += w;
	};
	auto add_entry_B = [&](int m, const Eigen::Vector3d& v) {
		B.row(m) += v;
	};

	for (int i = 0; i < num_V; ++i) {
		add_entry_A(i, i, 1);
		add_entry_B(i, Eigen::Vector3d(TV[i * 3],
			TV[i * 3 + 1], TV[i * 3 + 2]));
	}

	double regulation = 1.0;
	for (int i = 0; i < num_E; ++i) {
		for (int j = 0; j < 2; ++j) {
			int v0 = E[i * 2 + j];
			int v1 = E[i * 2 + 1 - j];
			double* V0 = V + v0 * 3;
			double* V1 = V + v1 * 3;
			Eigen::Vector3d off1(V1[0] - V0[0], V1[1] - V0[1], V1[2] - V0[2]);
			double reg = regulation * 2e-2 / off1.norm();
			off1 = scales[v0] * rotations[v0] * off1;
			add_entry_A(v0, v0, reg);
			add_entry_A(v0, v1, -reg);
			add_entry_A(v1, v0, -reg);
			add_entry_A(v1, v1, reg);
			add_entry_B(v0, -reg * off1);
			add_entry_B(v1, reg * off1);			
		}
	}

	typedef Eigen::Triplet<double> T;
	Eigen::SparseMatrix<double> A(num_entries, num_entries);
	std::vector<T> tripletList;
	for (auto& m : trips) {
		tripletList.push_back(T(m.first / num_entries,
			m.first % num_entries, m.second));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());

	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);

    solver.factorize(A);

    for (int j = 0; j < 3; ++j) {
        Eigen::VectorXd result = solver.solve(B.col(j));

        for (int i = 0; i < result.rows(); ++i) {
        	V[i * 3 + j] = result[i];
        }
    }	
}

typedef std::set<std::pair<int,int> >::iterator SIter;
typedef std::vector<std::pair<int,int> >::iterator VIter;

template void LinearEstimation<SIter>(std::vector<Vector3>& V,
	const std::vector<Eigen::Vector3i>& F,
	const SIter& E_begin, const SIter& E_end,
	const std::vector<int>& references,
	const std::vector<Vector3>& graphV,
	double rigidity2);

template void LinearEstimation<VIter>(std::vector<Vector3>& V,
	const std::vector<Eigen::Vector3i>& F,
	const VIter& E_begin, const VIter& E_end,
	const std::vector<int>& references,
	const std::vector<Vector3>& graphV,
	double rigidity2);
