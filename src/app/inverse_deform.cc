#include <chrono>
#include <iostream>

#include "deformer.h"
#include "mesh.h"
#include "uniformgrid.h"

// flags
int GRID_RESOLUTION = 64;
int MESH_RESOLUTION = 5000;

// main function
int main(int argc, char** argv) {	

	if (argc < 5) {
		printf("./inverse_deform source.obj reference.obj output.obj "
			"[GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] "
			"[lambda=1] [symmetry=0]\n");
		return 0;
	}

	//Deform source to fit the reference
	auto start = std::chrono::steady_clock::now();
	Mesh src, ref, cad;
	src.ReadOBJ(argv[1]);
	ref.ReadOBJ(argv[2]);

	int symmetry = 0;
	if (argc > 7) {
		sscanf(argv[7], "%d", &symmetry);
	}

	if (symmetry) {
		ref.ReflectionSymmetrize();
	}
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

	//ref.Normalize();
	//src.ApplyTransform(ref);

	printf("Reverse !\n");
	Deformer deform(lambda);
	deform.ReverseDeform(ref, &src);

	std::cout<<"Deformed"<<std::endl;

	src.WriteOBJ(argv[3]);
	auto end = std::chrono::steady_clock::now();

	std::cout << "Elapsed time in seconds : " 
		<< std::chrono::duration_cast
			<std::chrono::milliseconds>(end - start).count()
		<< " sec!\n";
	return 0;
}
