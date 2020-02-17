#include "meshcover.h"

#include <igl/point_mesh_squared_distance.h>

MeshCover::MeshCover()
{
	watertight = 0;
}

void MeshCover::Cover(const Mesh& watertight, Mesh& cad) {

	MatrixX V1(cad.V.size(), 3), V2(watertight.V.size(), 3);
	Eigen::MatrixXi F2(watertight.F.size(), 3);

	for (int i = 0; i < cad.V.size(); ++i)
		V1.row(i) = cad.V[i];

	for (int i = 0; i < watertight.V.size(); ++i)
		V2.row(i) = watertight.V[i];

	for (int i = 0; i < watertight.F.size(); ++i)
		F2.row(i) = watertight.F[i];

	MatrixX sqrD;
	Eigen::VectorXi I;
	MatrixX C;

	igl::point_mesh_squared_distance(V1,V2,F2,sqrD,I,C);

	findices.clear();
	weights.clear();
	for (int i = 0; i < I.size(); ++i) {
		Eigen::MatrixXd weight;
		int find = I[i];
		igl::barycentric_coordinates(C.row(i), V2.row(F2(find, 0)),
			V2.row(F2(find, 1)), V2.row(F2(find, 2)), weight);
		findices.push_back(find);
		weights.push_back(weight.row(0));
	}
	cover = cad;

	this->watertight = &watertight;
}

void MeshCover::UpdateCover() {
	if (!watertight)
		return;
	for (int i = 0; i < findices.size(); ++i) {
		auto& f = watertight->F[findices[i]];
		const Vector3& v0 = watertight->V[f[0]];
		const Vector3& v1 = watertight->V[f[1]];
		const Vector3& v2 = watertight->V[f[2]];
		Vector3 v = v0 * weights[i][0] + v1 * weights[i][1] + v2 * weights[i][2];
		cover.V[i] = v;
	}
}
