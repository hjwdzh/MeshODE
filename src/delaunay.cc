#include "delaunay.h"

#include "types.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>

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

template < class GT, class Vb = CGAL::Triangulation_vertex_base_3<GT> >
class My_vertex_base
  : public Vb
{
public:
  typedef typename Vb::Vertex_handle  Vertex_handle;
  typedef typename Vb::Cell_handle    Cell_handle;
  typedef typename Vb::Point          Point;
  template < class TDS2 >
  struct Rebind_TDS {
    typedef typename Vb::template Rebind_TDS<TDS2>::Other  Vb2;
    typedef My_vertex_base<GT, Vb2>                        Other;
  };
  My_vertex_base() {}
  My_vertex_base(const Point& p) : Vb(p) {}
  My_vertex_base(const Point& p, Cell_handle c) : Vb(p, c) {}
  Vertex_handle   vh;
  Cell_handle     ch;
};

typedef CGAL::Delaunay_triangulation_cell_base_3<K>                  Cb;
typedef CGAL::Triangulation_data_structure_3<My_vertex_base<K>, Cb>  Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds>                       Delaunay3D;
typedef Delaunay::Vertex_handle                                      Vertex_handle;
typedef Delaunay::Point                                              Point3D;

void delaunay3d(Eigen::MatrixXd& V, std::vector<std::pair<int, int> >& edges)
{
	std::vector<Point3D> points(V.rows());
	std::map<std::pair<int, std::pair<int, int> >, int > p2k;
	auto make_key = [&](const Point3D& p) {
		return std::make_pair(p.x() * 1e7, std::make_pair(p.y() * 1e7, p.z() * 1e7));
	};
	for (int i = 0; i < V.rows(); ++i) {
		points[i] = Point(V(i, 0), V(i, 1), V(i, 2));
		p2k[make_key(points[i])] = i;
	}

	Delaunay3D T;
	T.insert(points.begin(), points.end());

	for(auto f = T.finite_edges_begin(); f != T.finite_edges_end(); ++f) {
    	auto seg = T.segment( *f );
    	int v1 = p2k[make_key(seg.point(0))];
    	int v2 = p2k[make_key(seg.point(1))];
    	edges.push_back(std::make_pair(v1, v2));
	}

}

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_conformer_2.h>

#include <cassert>
#include <iostream>
#include <fstream>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tdss;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tdss> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Vertex_handle Vertex_handle_CDT;
typedef CDT::Point Point_CDT;

void delaunay_with_boundary(Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::vector<std::pair<int, int> >& boundary_edges)
{
  CDT cdt;

	std::vector<Point_CDT> points(V.rows());
	std::map<std::pair<int, int>, int > p2k;
	auto make_key = [&](const Point_CDT& p) {
		printf("<%f %f>\n", p.x(), p.y());
		return std::make_pair(p.x() * 1e7, p.y() * 1e7);
	};
	for (int i = 0; i < V.rows(); ++i) {
		points[i] = Point_CDT(V(i, 0), V(i, 1));
		p2k[make_key(points[i])] = i;
	}

  double min_x = 1e30;
  std::vector<int> is_boundary(V.rows(), 0);
  for (auto& b : boundary_edges) {
  	is_boundary[b.first] = 1;
  	is_boundary[b.second] = 1;
  	auto v0 = V.row(b.first);
  	auto v1 = V.row(b.second);
  	cdt.insert_constraint( Point_CDT(v0[0], v0[1]), Point_CDT(v1[0], v1[1]));
  }

  std::vector<Point_CDT> p;
  for (int i = 0; i < is_boundary.size(); ++i) {
  	if (V(i, 0) < min_x)
  		min_x = V(i, 0);
  	if (is_boundary[i] == 0) {
  		p.push_back(Point_CDT(V(i, 0), V(i, 1)));
  	}
  }

  cdt.insert(p.begin(), p.end());

  std::list<Point_CDT> list_of_seeds;
  list_of_seeds.push_back(Point_CDT(min_x - 1.0, 0));

  CGAL::refine_Delaunay_mesh_2(cdt, list_of_seeds.begin(), list_of_seeds.end(),
    Criteria());

	int face_count = 0;
	for(auto f : cdt.finite_face_handles()) {
		if (!f->is_in_domain())
			continue;
		face_count += 1;
	}
	F = Eigen::MatrixXi(face_count, 3);
	int top = 0;
	for(auto f : cdt.finite_face_handles()) {
		if (!f->is_in_domain())
			continue;
	    for (int j = 0; j < 3; ++j) {
	    	auto v = f->vertex(j)->point();
	    	F(top, j) = p2k[make_key(v)];
	    }
	    top += 1;
	}
}