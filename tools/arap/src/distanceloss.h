#ifndef ARAP_DISTANCELOSS_H_
#define ARAP_DISTANCELOSS_H_

#include "uniformgrid.h"

struct DistanceLoss {
  DistanceLoss(UniformGrid* grid_)
  : grid(grid_) {}

  template <typename T>
  bool operator()(const T* const p1,
                  T* residuals) const {
  	residuals[0] = grid->distance(p1);
  	residuals[1] = (T)0;
  	residuals[2] = (T)0;
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(UniformGrid* grid) {
     return (new ceres::AutoDiffCostFunction<DistanceLoss, 3, 3>(
                 new DistanceLoss(grid)));
   }
   UniformGrid* grid;
};

#endif