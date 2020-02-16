#ifndef ARAP_UNIFORM_GRID_H_
#define ARAP_UNIFORM_GRID_H_

#include <vector>

#include "types.h"

class UniformGrid
{
public:
	UniformGrid();
	UniformGrid(int _N);
	template <class T>
	T distance(const T* const p) const;

	int N;
	std::vector<std::vector<std::vector<FT> > > distances;
};

#endif