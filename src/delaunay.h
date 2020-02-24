#ifndef ARAP_DELAUNAY_H_
#define ARAP_DELAUNAY_H_

#include <Eigen/Core>

void delaunay(Eigen::MatrixXd& V, Eigen::MatrixXi& F);
void delaunay3d(Eigen::MatrixXd& V, std::vector<std::pair<int, int> >& edges);

void delaunay_with_boundary(Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::vector<std::pair<int, int> >& boundary_edges);

#endif