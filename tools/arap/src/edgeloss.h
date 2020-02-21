#ifndef ARAP_EDGELOSS_H_
#define ARAP_EDGELOSS_H_

#include "types.h"

#include <ceres/rotation.h>

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

struct AdaptiveEdgeLoss {
  AdaptiveEdgeLoss(const Vector3& v_, FT lambda_)
  : v(v_), lambda(lambda_) {
    lambda = lambda * (2e-2 / v.norm());
  }

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
     return (new ceres::AutoDiffCostFunction<AdaptiveEdgeLoss, 3, 3, 3>(
                 new AdaptiveEdgeLoss(v, lambda_)));
   }
   Vector3 v;
   FT lambda;
};

struct EdgeLossWithRot {
  EdgeLossWithRot(const Vector3& v_, FT lambda_)
  : v(v_), lambda(lambda_) {}

  template <typename T>
  bool operator()(const T* const p1,
                  const T* const p2,
                  const T* const rot1,
                  const T* const rot2,
                  T* residuals) const {

    T p[3], q[3];
    p[0] = p1[0] - p2[0];
    p[1] = p1[1] - p2[1];
    p[2] = p1[2] - p2[2];
    ceres::AngleAxisRotatePoint(rot1, p, q);
    residuals[0] = (q[0] - (T)v[0]) * (T)lambda;
    residuals[1] = (q[1] - (T)v[1]) * (T)lambda;
    residuals[2] = (q[2] - (T)v[2]) * (T)lambda;
    residuals[3] = (rot1[0] - rot2[0]) * (T)1;
    residuals[4] = (rot1[1] - rot2[1]) * (T)1;
    residuals[5] = (rot1[2] - rot2[2]) * (T)1;
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const Vector3& v, const FT lambda_) {
     return (new ceres::AutoDiffCostFunction<EdgeLossWithRot, 6, 3, 3, 3, 3>(
                 new EdgeLossWithRot(v, lambda_)));
   }
   Vector3 v;
   FT lambda;
};

#endif