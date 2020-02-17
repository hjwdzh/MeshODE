#include "meshcover.h"

#include <igl/point_mesh_squared_distance.h>

MeshCover::MeshCover()
{}

void MeshCover::Cover(const Mesh& watertight, const Mesh& cad) {
	cover = watertight;

	MatrixX V1(watertight.V.size(), 3), V2(cad.V.size(), 3);
	Eigen::MatrixXi F2(cad.F.size(), 3);

	for (int i = 0; i < watertight.V.size(); ++i)
		V1.row(i) = watertight.V[i];

	for (int i = 0; i < cad.V.size(); ++i)
		V2.row(i) = cad.V[i];

	for (int i = 0; i < cad.F.size(); ++i)
		F2.row(i) = cad.F[i];

	MatrixX sqrD;
	Eigen::VectorXi I;
	MatrixX C;

	igl::point_mesh_squared_distance(V1,V2,F2,sqrD,I,C);

	for (int i = 0; i < cover.V.size(); ++i)
		cover.V[i] = C.row(i);
}
