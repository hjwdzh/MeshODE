#ifndef SHAPEDEFORM_LINEAR_H_
#define SHAPEDEFORM_LINEAR_H_

#include "types.h"

template<class Iter>
void LinearEstimation(std::vector<Vector3>& V,
	const std::vector<Eigen::Vector3i>& F,
	const Iter& E_begin, const Iter& E_end,
	const std::vector<int>& references,
	const std::vector<Vector3>& graphV,
	double rigidity = 2.0);

void LinearEstimationWithRot(double* V, int* F, double* TV,
	int num_V, int num_F, double rigidity);

#endif