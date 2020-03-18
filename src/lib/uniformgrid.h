#ifndef SHAPEDEFORM_UNIFORM_GRID_H_
#define SHAPEDEFORM_UNIFORM_GRID_H_

#include <vector>

#include "types.h"

class UniformGrid
{
public:
	UniformGrid();
	UniformGrid(int grid_dimension);
	
	template <class T>
	T distance(const T* const p) const;

	template <class T>
	T DistanceFloat(const T* const p) const;

	int Dimension() const {
		return grid_dimension_;
	}

	void SetDistance(int i, int j, int k, FT distance) {
		voxel_distance_[i][j][k] = distance;
	}

	FT GetDistance(int i, int j, int k) const {
		return voxel_distance_[i][j][k];
	}
private:
	int grid_dimension_;
	std::vector<std::vector<std::vector<FT> > > voxel_distance_;
};

#endif