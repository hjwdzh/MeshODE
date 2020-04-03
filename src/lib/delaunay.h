#ifndef SHAPEDEFORM_DELAUNAY_H_
#define SHAPEDEFORM_DELAUNAY_H_

#include <vector>
#include <Eigen/Core>

void Delaunay2D(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

void Delaunay3D(Eigen::MatrixXd& V, std::vector<std::pair<int, int> >& edges);

#endif