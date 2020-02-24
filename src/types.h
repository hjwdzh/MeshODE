#ifndef ARAP_TYPES_H_
#define ARAP_TYPES_H_

#include <Eigen/Core>
#include <Eigen/Sparse>

typedef double FT;
typedef Eigen::Matrix<FT, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
typedef Eigen::Matrix<FT, Eigen::Dynamic, 1> VectorX;
typedef Eigen::SparseMatrix<FT> SpMat;
typedef Eigen::Triplet<FT> T;
typedef Eigen::Matrix<FT, 3, 1> Vector3;

#endif