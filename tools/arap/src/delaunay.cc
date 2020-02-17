#include "delaunay.h"

#include "types.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <iostream>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Projection_traits_xy_3<K>  Gt;
typedef CGAL::Delaunay_triangulation_2<Gt> Delaunay;
typedef K::Point_3   Point;

void delaunay(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	std::vector<Point> points(V.rows());
	std::map<std::pair<int, int>, int > p2k;
	auto make_key = [&](Point& p) {
		return std::make_pair(p.x() * 1e7, p.y() * 1e7);
	};
	for (int i = 0; i < V.rows(); ++i) {
		points[i] = Point(V(i, 0), V(i, 1), 0);
		p2k[make_key(points[i])] = i;
	}

	Delaunay T;
	T.insert(points.begin(), points.end());
	int face_count = 0;
	for(auto f : T.finite_face_handles())
		face_count += 1;
	F = Eigen::MatrixXi(face_count, 3);
	int top = 0;
	for(auto f : T.finite_face_handles()) {
	    for (int j = 0; j < 3; ++j) {
	    	auto v = f->vertex(j)->point();
	    	F(top, j) = p2k[make_key(v)];
	    }
	    top += 1;
	}

}
