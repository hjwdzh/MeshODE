#include <fstream>
#include <iostream>

#include "callback.h"
#include "deformer.h"
#include "mesh.h"
#include "subdivision.h"
#include "uniformgrid.h"

// flags
int GRID_RESOLUTION = 64;
int MESH_RESOLUTION = 5000;

std::vector<Vector3>* vertex_pointer;
std::vector<std::vector<Vector3> > flow;
void callback()
{
	flow.push_back(*vertex_pointer);
}
// main function
int main(int argc, char** argv) {	
	if (argc < 5) {
		printf("./cad_deform cad.obj reference.obj output.obj "
			"[GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] "
			"[lambda=1] [symmetry=0] [flow_output=filename]\n");
		return 0;
	}
	//Deform source to fit the reference


	Mesh ref, cad;
	cad.ReadOBJ(argv[1]);
	ref.ReadOBJ(argv[2]);

	int symmetry = 0;
	if (argc > 7) {
		sscanf(argv[7], "%d", &symmetry);
	}

	if (symmetry)
		ref.ReflectionSymmetrize();

	cad.RemoveDegenerated();
	cad.MergeDuplex();
	//cad.WriteOBJ("../example/test.obj");

	Subdivision sub;
	sub.Subdivide(cad, 2e-2);

	sub.ComputeGeometryNeighbors(3e-2);

	//sub.ComputeGeometryNeighbors(1e-2);
	if (argc > 4)
		sscanf(argv[4], "%d", &GRID_RESOLUTION);

	if (argc > 5)
		sscanf(argv[5], "%d", &MESH_RESOLUTION);

	FT lambda = 1;
	if (argc > 6)
		sscanf(argv[6], "%lf", &lambda);

	//Get number of vertices and faces
	std::cout << "Source:\t\t" << "Num vertices: " << cad.GetV().size()
		<< "\tNum faces: " << cad.GetF().size() << std::endl;
	std::cout<<"Reference:\t" << "Num vertices: " <<ref.GetV().size()
		<< "\tNum faces: " << ref.GetF().size() <<std::endl <<std::endl;

	UniformGrid grid(GRID_RESOLUTION);
	ref.Normalize();
	auto& subdivide_mesh = sub.GetMesh();
	subdivide_mesh.ApplyTransform(ref);
	ref.ConstructDistanceField(grid);

	int need_callback = 0;
	std::string flow_file = "";
	if (argc > 8) {
		need_callback = 1;
		flow_file = argv[8];
	}

	if (!need_callback) {
		Deformer deformer(lambda);
		deformer.DeformSubdivision(grid, &sub);
	} else {
		Deformer deformer(lambda, callback);

		vertex_pointer = &subdivide_mesh.GetV();
		deformer.DeformSubdivision(grid, &sub);
		flow.push_back(*vertex_pointer);

		std::ofstream os(flow_file);
		int top = 1;
		
		FT scale = subdivide_mesh.GetScale();
		Vector3 pos = subdivide_mesh.GetTranslation();

		for (int i = 0; i < flow[0].size(); ++i) {
			for (int j = 0; j < flow.size(); ++j) {
				auto v = flow[j][i] * scale + pos;
				os << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
			}
			for (int j = 0; j < flow.size() - 1; ++j) {
				os << "l " << top + j << " " << top + j + 1 << "\n";
			}
			top += flow.size();
		}
		os.close();
	}
	std::cout<<"Deformed"<<std::endl;

	subdivide_mesh.WriteOBJ(argv[3]);
	return 0;
}
