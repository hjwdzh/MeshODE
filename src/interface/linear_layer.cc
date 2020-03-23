#include "linear_layer.h"

#include <linear.h>

void SolveLinear(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	torch::Tensor tensorE,
	torch::Tensor tensorRef,
	torch::Tensor tensorGraphV,
	double rigidity) {

	std::vector<Vector3> V;
	std::vector<Eigen::Vector3i> F;
	std::vector<std::pair<int, int> > E;

	std::vector<int> references;
	std::vector<Vector3> graphV;

#ifndef USE_DOUBLE
	typedef float T;
#else
	typedef double T;
#endif
	int v_size = tensorV.size(0);
	auto dataV = static_cast<T*>(tensorV.storage().data());
	V.resize(v_size);
	for (int i = 0; i < v_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			V[i][j] = dataV[i * 3 + j];
		}
	}

	int f_size = tensorF.size(0);
	auto dataF = static_cast<int*>(tensorF.storage().data());
	F.resize(f_size);
	for (int i = 0; i < f_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			F[i][j] = dataF[i * 3 + j];
		}
	}

	int e_size = tensorE.size(0);
	auto dataE = static_cast<int*>(tensorE.storage().data());
	E.resize(e_size);
	for (int i = 0; i < e_size; ++i) {
		E[i].first = dataE[i * 2];
		E[i].second = dataE[i * 2 + 1];
	}

	int r_size = tensorRef.size(0);
	auto dataRef = static_cast<int*>(tensorRef.storage().data());
	references.resize(r_size);
	for (int i = 0; i < r_size; ++i) {
		references[i] = dataRef[i];
	}

	int graphV_size = tensorGraphV.size(0);
	auto dataGraphV = static_cast<T*>(tensorGraphV.storage().data());
	graphV.resize(graphV_size);
	for (int i = 0; i < graphV_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			graphV[i][j] = dataGraphV[i * 3 + j];
		}
	}

	LinearEstimation(V, F, E.begin(), E.end(), references, graphV, rigidity);
	for (int i = 0; i < v_size; ++i) {
		for (int j = 0; j < 3; ++j) {
			dataV[i * 3 + j] = V[i][j];
		}
	}
}

