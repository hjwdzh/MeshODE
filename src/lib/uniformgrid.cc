#include "uniformgrid.h"

#include <ceres/ceres.h>

UniformGrid::UniformGrid()
: grid_dimension_(0)
{}

UniformGrid::UniformGrid(int grid_dimension) {
	grid_dimension_ = grid_dimension;
	voxel_distance_.resize(grid_dimension);
	for (auto& d : voxel_distance_) {
		d.resize(grid_dimension);
		for (auto& v : d)
			v.resize(grid_dimension, 1e30);
	}
}
template <class T>
T UniformGrid::distance(const T* const p) const {
	int px = *(double*)&p[0] * grid_dimension_;
	int py = *(double*)&p[1] * grid_dimension_;
	int pz = *(double*)&p[2] * grid_dimension_;
	if (px < 0 || py < 0 || pz < 0
		|| px >= grid_dimension_ - 1
		|| py >= grid_dimension_ - 1
		|| pz >= grid_dimension_ - 1) {

		T l = (T)0;
		if (px < 0)
			l = l + -p[0] * (T)grid_dimension_;
		else if (px >= grid_dimension_)
			l = l + (p[0] * (T)grid_dimension_
				- (T)(grid_dimension_ - 1 - 1e-3));

		if (py < 0)
			l = l + -p[1] * (T)grid_dimension_;
		else if (py >= grid_dimension_)
			l = l + (p[1] * (T)grid_dimension_
				- (T)(grid_dimension_ - 1 - 1e-3));

		if (pz < 0)
			l = l + -p[2] * (T)grid_dimension_;
		else if (pz >= grid_dimension_)
			l = l + (p[2] * (T)grid_dimension_
				- (T)(grid_dimension_ - 1 - 1e-3));

		return l;

	}
	T wx = p[0] * (T)grid_dimension_ - (T)px;
	T wy = p[1] * (T)grid_dimension_ - (T)py;
	T wz = p[2] * (T)grid_dimension_ - (T)pz;

	T w0 = ((T)1 - wx) * ((T)1 - wy) * ((T)1 - wz) *
			(T)voxel_distance_[pz    ][py    ][px    ];

	T w1 = wx 		   * ((T)1 - wy) * ((T)1 - wz) *
			(T)voxel_distance_[pz    ][py    ][px + 1];

	T w2 = ((T)1 - wx) * wy 		 * ((T)1 - wz) *
			(T)voxel_distance_[pz    ][py + 1][px    ];

	T w3 = wx 		   * wy 		 * ((T)1 - wz) *
			(T)voxel_distance_[pz    ][py + 1][px + 1];

	T w4 = ((T)1 - wx) * ((T)1 - wy) * wz 		   *
			(T)voxel_distance_[pz + 1][py    ][px    ];

	T w5 = wx 		   * ((T)1 - wy) * wz 		   *
			(T)voxel_distance_[pz + 1][py    ][px + 1];

	T w6 = ((T)1 - wx) * wy 		 * wz		   *
			(T)voxel_distance_[pz + 1][py + 1][px    ];

	T w7 = wx 		   * wy 		 * wz 		   *
			(T)voxel_distance_[pz + 1][py + 1][px + 1];

	T res = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7;

	if (res > 0.2)
		return T(0);
	return res;
}

template FT UniformGrid::distance<FT>(const FT* const) const;
template ceres::Jet<double, 3> UniformGrid::distance<ceres::Jet<double, 3> >(
	const ceres::Jet<double, 3>* const) const;
