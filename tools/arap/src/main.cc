#include <iostream>

#include "deform.h"
#include "mesh.h"
#include "meshcover.h"
#include "uniformgrid.h"

// flags
int GRID_RESOLUTION = 64;
int MESH_RESOLUTION = 5000;

// main function
int main(int argc, char** argv) {	
	if (argc < 5) {
		printf("./deform source.obj reference.obj cad.obj output.obj [GRID_RESOLUTION=64] [MESH_RESOLUTION=5000]\n");
		return 0;
	}
	//Deform source to fit the reference

	Mesh src, ref, cad;
	src.ReadOBJ(argv[1]);
	ref.ReadOBJ(argv[2]);
	cad.ReadOBJ(argv[3]);

	//MeshCover shell;
	//shell.Cover(src, cad);

	//shell.cover.WriteOBJ("debug.obj");

	//src = shell.cover;
	//return 0;
	if (argc > 5)
		sscanf(argv[5], "%d", &GRID_RESOLUTION);

	if (argc > 6)
		sscanf(argv[6], "%d", &MESH_RESOLUTION);

	FT lambda = 1;
	if (argc > 7)
		sscanf(argv[7], "%lf", &lambda);
	printf("lambda %lf\n", lambda);
	//Get number of vertices and faces
	std::cout<<"Source:\t\t"<<"Num vertices: "<<src.V.size()<<"\tNum faces: "<<src.F.size()<<std::endl;
	std::cout<<"Reference:\t"<<"Num vertices: "<<ref.V.size()<<"\tNum faces: "<<ref.F.size()<<std::endl<<std::endl;

	UniformGrid grid(GRID_RESOLUTION);
	ref.Normalize();
	src.ApplyTransform(ref);

	ref.ConstructDistanceField(grid);
	//src.HierarchicalDeform(grid);
	DeformWithRot(src, grid, lambda);

	std::cout<<"Deformed"<<std::endl;

	src.WriteOBJ(argv[4]);
	return 0;
}
