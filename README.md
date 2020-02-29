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
```
export LD_LIBRARY_PATH=/orion/u/jingweih/3rd/cgal-install/lib:/orion/u/jingweih/3rd/ceres-solver/install/lib
./deform ./source.obj ./reference.obj output.obj 64 5000 10
```
10 is the rigidity strength

