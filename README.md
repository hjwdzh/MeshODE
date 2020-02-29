# Deep Shape Deformation.
Deform Shape A to fit shape B.

![Plane Fitting Results](https://github.com/hjwdzh/ShapeDeform/raw/master/res/teaser.png)

### Dependencies
1. libIGL
2. CGAL
3. Ceres

### Build
```
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
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
${binary} source.obj reference.obj output.obj [GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] [lambda=1] [symmetry=0].
```


