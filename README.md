# Deep Shape Deformation.
Deform Shape A to fit shape B.

![Plane Fitting Results](https://github.com/hjwdzh/ShapeDeform/raw/master/res/teaser.png)

### Dependencies
1. libIGL
2. CGAL
3. Ceres
4. pytorch

### Setup CMake Example
```
cmake .. -DCMAKE_BUILD_TYPE=Release -Dceres_DIR=/orion/u/jingweih/3rd/ceres-solver/install/lib/cmake/Ceres -DCGAL_INCLUDE_PATH=/orion/u/jingweih/3rd/cgal-install/include -DEIGEN_INCLUDE_PATH=/orion/u/jingweih/3rd/eigen3 -DIGL_INCLUDE_PATH=/orion/u/jingweih/3rd -DTORCH_PATH=/orion/u/jingweih/pytorch/lib/python3.5/site-packages/torch
```

### Build
```
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release -DIGL_INCLUDE_PATH= -DTORCH_PATH=XXX
make -j8
```

### Download test data
```
cd data
sh download.sh
```

### Run
We provide different binaries for shape deformation with different assumptions.
1. rigid_deform.
	Deform well-connected and uniform triangle meshes A to general shape B so that regions in A are close to B.
2. rigid_rot_deform.
	Similar to rigid_deform. Preserving edge length instead of 3D offset, which fits better but potentially more distortion.
3. cad_deform.
	Deform a CAD model without preassumption of connectivity or uniformness, using rigid_deform.
4. coverage_deform. (experimental)
	Deform A to B in order to cover most regions in B without distorting A too much.
5. inverse_deform. (experimental)
	Deform A to B so that regions in B are close to A.

The way to run them is by
```
./rigid_deform source.obj reference.obj output.obj [GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] [lambda=1] [symmetry=0].
```

### Run Pytorch optimizer
```
cd build
export PYTHONPATH=$PYTHONPATH:$(pwd)
python ../src/python/rigid_deform.py ../data/source.obj ../data/reference.obj ./rigid_output.obj
python ../src/python/cad_deform2.py ../data/cad.obj ../data/reference.obj ./cad_output.obj 0.3
```

