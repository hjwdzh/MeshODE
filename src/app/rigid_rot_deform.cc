#include <iostream>

#include "deformer.h"
#include "mesh.h"
#include "meshcover.h"
#include "uniformgrid.h"

// flags
int GRID_RESOLUTION = 64;
int MESH_RESOLUTION = 5000;

// main function
int main(int argc, char** argv) {	
	if (argc < 5) {
		printf("./rigid_rot_deform source.obj reference.obj output.obj "
			"[GRID_RESOLUTION=64] [MESH_RESOLUTION=5000]\n");
		return 0;
	}

	//Deform source to fit the reference
	Mesh src, ref;
	src.ReadOBJ(argv[1]);
	ref.ReadOBJ(argv[2]);

	if (argc > 4)
		sscanf(argv[4], "%d", &GRID_RESOLUTION);

	if (argc > 5)
		sscanf(argv[5], "%d", &MESH_RESOLUTION);

	FT lambda = 1;
	if (argc > 6)
		sscanf(argv[6], "%lf", &lambda);

	//Get number of vertices and faces
	std::cout << "Source:\t\t" << "Num vertices: " << src.GetV().size()
		<< "\tNum faces: " << src.GetF().size() << std::endl;
	std::cout<<"Reference:\t" << "Num vertices: " <<ref.GetV().size()
		<< "\tNum faces: " << ref.GetF().size() <<std::endl <<std::endl;

	UniformGrid grid(GRID_RESOLUTION);
	ref.Normalize();
	src.ApplyTransform(ref);

	ref.ConstructDistanceField(grid);
	//src.HierarchicalDeform(grid);

	Deformer deform(lambda);
	deform.DeformWithRot(grid, &src);

	std::cout<<"Deformed"<<std::endl;

	src.WriteOBJ(argv[3]);
	return 0;
}
