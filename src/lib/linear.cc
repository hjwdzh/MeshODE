#include "linear.h"

#include <set>
#include <unordered_map>

template<class Iter>
void LinearEstimation(std::vector<Vector3>& V,
	const std::vector<Eigen::Vector3i>& F,
	const Iter& E_begin, const Iter& E_end,
	const std::vector<int>& references,
	const std::vector<Vector3>& graphV)
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


	FT regular = 2;

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

	printf("Linear Solve...\n");
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

typedef std::set<std::pair<int,int> >::iterator SIter;
typedef std::vector<std::pair<int,int> >::iterator VIter;

template void LinearEstimation<SIter>(std::vector<Vector3>& V,
	const std::vector<Eigen::Vector3i>& F,
	const SIter& E_begin, const SIter& E_end,
	const std::vector<int>& references,
	const std::vector<Vector3>& graphV);

template void LinearEstimation<VIter>(std::vector<Vector3>& V,
	const std::vector<Eigen::Vector3i>& F,
	const VIter& E_begin, const VIter& E_end,
	const std::vector<int>& references,
	const std::vector<Vector3>& graphV);
