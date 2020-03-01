# Deep Shape Deformation.
Deform Shape A to fit shape B.

![Plane Fitting Results](https://github.com/hjwdzh/ShapeDeform/raw/master/res/teaser.png)

### Dependencies
1. libIGL
2. CGAL
3. Ceres
4. pybind11

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
python ../src/pytorch/optimize_rigid_deform.py ../data/source.obj ../data/reference.obj ./output.obj
```

