#include "uniformgrid.h"

#include <ceres/ceres.h>

UniformGrid::UniformGrid()
: N(0)
{}

UniformGrid::UniformGrid(int _N) {
	N = _N;
	distances.resize(N);
	for (auto& d : distances) {
		d.resize(N);
		for (auto& v : d)
			v.resize(N, 1e30);
	}
}
template <class T>
T UniformGrid::distance(const T* const p) const {
	int px = *(double*)&p[0] * N;
	int py = *(double*)&p[1] * N;
	int pz = *(double*)&p[2] * N;
	if (px < 0 || py < 0 || pz < 0 || px >= N - 1 || py >= N - 1 || pz >= N - 1) {
		T l = (T)0;
		if (px < 0)
			l = l + -p[0] * (T)N;
		else if (px >= N)
			l = l + (p[0] * (T)N - (T)(N - 1 - 1e-3));

		if (py < 0)
			l = l + -p[1] * (T)N;
		else if (py >= N)
			l = l + (p[1] * (T)N - (T)(N - 1 - 1e-3));

		if (pz < 0)
			l = l + -p[2] * (T)N;
		else if (pz >= N)
			l = l + (p[2] * (T)N - (T)(N - 1 - 1e-3));

		return l;
	}
	T wx = p[0] * (T)N - (T)px;
	T wy = p[1] * (T)N - (T)py;
	T wz = p[2] * (T)N - (T)pz;
	T w0 = ((T)1 - wx) * ((T)1 - wy) * ((T)1 - wz) * (T)distances[pz    ][py    ][px    ];
	T w1 = wx 		   * ((T)1 - wy) * ((T)1 - wz) * (T)distances[pz    ][py    ][px + 1];
	T w2 = ((T)1 - wx) * wy 		 * ((T)1 - wz) * (T)distances[pz    ][py + 1][px    ];
	T w3 = wx 		   * wy 		 * ((T)1 - wz) * (T)distances[pz    ][py + 1][px + 1];
	T w4 = ((T)1 - wx) * ((T)1 - wy) * wz 		   * (T)distances[pz + 1][py    ][px    ];
	T w5 = wx 		   * ((T)1 - wy) * wz 		   * (T)distances[pz + 1][py    ][px + 1];
	T w6 = ((T)1 - wx) * wy 		 * wz		   * (T)distances[pz + 1][py + 1][px    ];
	T w7 = wx 		   * wy 		 * wz 		   * (T)distances[pz + 1][py + 1][px + 1];
	T res = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7;

	return res;
}

template FT UniformGrid::distance<FT>(const FT* const) const;
template ceres::Jet<double, 3> UniformGrid::distance<ceres::Jet<double, 3> >(const ceres::Jet<double, 3>* const) const;
