#ifndef ARAP_EDGELOSS_H_
#define ARAP_EDGELOSS_H_

#include "types.h"

struct EdgeLoss {
  EdgeLoss(const Vector3& v_, FT lambda_)
  : v(v_), lambda(lambda_) {}

  template <typename T>
  bool operator()(const T* const p1,
                  const T* const p2,
                  T* residuals) const {
  	T px = p1[0] - p2[0];
  	T py = p1[1] - p2[1];
  	T pz = p1[2] - p2[2];
  	residuals[0] = (px - (T)v[0]) * (T)lambda;
  	residuals[1] = (py - (T)v[1]) * (T)lambda;
  	residuals[2] = (pz - (T)v[2]) * (T)lambda;
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const Vector3& v, const FT lambda_) {
     return (new ceres::AutoDiffCostFunction<EdgeLoss, 3, 3, 3>(
                 new EdgeLoss(v, lambda_)));
   }
   Vector3 v;
   FT lambda;
};

#endif