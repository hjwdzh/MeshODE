#include <fstream>
#include <iostream>

#include "deform.h"
#include "mesh.h"
#include "meshcover.h"
#include "subdivision.h"
#include "uniformgrid.h"

// flags
int GRID_RESOLUTION = 64;
int MESH_RESOLUTION = 5000;

// main function
int main(int argc, char** argv) {	
	if (argc < 5) {
		printf("./deform cad.obj reference.obj output.obj [GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] [lambda=1] [symmetry=0]\n");
		return 0;
	}
	//Deform source to fit the reference


	Mesh ref, cad;
	cad.ReadOBJ(argv[1]);
	ref.ReadOBJ(argv[2]);

	int symmetry = 0;
	if (argc > 7) {
		sscanf(argv[6], "%d", &symmetry);
	}

	if (symmetry)
		ref.ReflectionSymmetrize();

	Subdivision sub;
	sub.Subdivide(cad, 1e-2);
	sub.subdivide_mesh.MergeDuplex();

	sub.ComputeGeometryNeighbors(1.5e-2);

	//sub.ComputeGeometryNeighbors(1e-2);
	if (argc > 4)
		sscanf(argv[4], "%d", &GRID_RESOLUTION);

	if (argc > 5)
		sscanf(argv[5], "%d", &MESH_RESOLUTION);

	FT lambda = 1;
	if (argc > 6)
		sscanf(argv[6], "%lf", &lambda);
	printf("lambda %lf\n", lambda);
	//Get number of vertices and faces
	std::cout<<"Source:\t\t"<<"Num vertices: "<<cad.V.size()<<"\tNum faces: "<<cad.F.size()<<std::endl;
	std::cout<<"Reference:\t"<<"Num vertices: "<<ref.V.size()<<"\tNum faces: "<<ref.F.size()<<std::endl<<std::endl;

	UniformGrid grid(GRID_RESOLUTION);
	ref.Normalize();
	sub.subdivide_mesh.ApplyTransform(ref);
	ref.ConstructDistanceField(grid);
	//sub.subdivide_mesh.WriteOBJ("debug.obj");
	//ref.WriteOBJ("debug1.obj");
	//src.HierarchicalDeform(grid);
	//Deform(src, grid, lambda);
	DeformSubdivision(sub, grid, lambda);
	std::cout<<"Deformed"<<std::endl;

	//cover.UpdateCover(src, sub);
	sub.subdivide_mesh.WriteOBJ(argv[3]);
	return 0;
}
