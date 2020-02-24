#ifndef ARAP_DISTANCELOSS_H_
#define ARAP_DISTANCELOSS_H_

#include "uniformgrid.h"

struct DistanceLoss {
	DistanceLoss(const UniformGrid* grid_)
	: grid(grid_) {}

	template <typename T>
	bool operator()(const T* const p1, T* residuals) const {
		residuals[0] = grid->distance(p1);
		residuals[1] = (T)0;
		residuals[2] = (T)0;
		return true;
	}

	 // Factory to hide the construction of the CostFunction object from
	 // the client code.
	 static ceres::CostFunction* Create(const UniformGrid* grid) {
		 return (new ceres::AutoDiffCostFunction<DistanceLoss, 3, 3>(
								 new DistanceLoss(grid)));
	 }
	 const UniformGrid* grid;
};

struct BarycentricDistanceLoss {
	BarycentricDistanceLoss(const Vector3& w_, const Vector3& tar_)
	: w(w_), tar(tar_) {}

	template <typename T>
	bool operator()(const T* const p1, const T* const p2, const T* const p3,
		T* residuals) const {
		for (int j = 0; j < 3; ++j) {
			residuals[j] = (T)w[0] * p1[j]
						 + (T)w[1] * p2[j]
						 + (T)w[2] * p3[j] - (T)tar[j];
		}
		return true;
	}

	static ceres::CostFunction* Create(const Vector3& w_, const Vector3& tar_) {
		 return (new ceres::AutoDiffCostFunction<
					BarycentricDistanceLoss, 3, 3, 3, 3>(
			new BarycentricDistanceLoss(w_, tar_)));

	}
	Vector3 w, tar;
};

struct PointRegularizerLoss {
	PointRegularizerLoss(const double lambda_, const Vector3& tar_)
	: lambda(lambda_), tar(tar_) {}

	template <typename T>
	bool operator()(const T* const p, T* residuals) const {
		for (int j = 0; j < 3; ++j) {
			residuals[j] = (T)lambda * (p[j] - (T)tar[j]);
		}
		return true;
	}

	static ceres::CostFunction* Create(
		const double lambda_, const Vector3& tar_) {
		 return (new ceres::AutoDiffCostFunction<PointRegularizerLoss, 3, 3>(
								 new PointRegularizerLoss(lambda_, tar_)));

	}
	double lambda;
	Vector3 tar;
};

#endif