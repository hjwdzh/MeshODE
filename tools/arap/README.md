# ARAP

### Build
```
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dceres_DIR=/orion/u/jingweih/3rd/ceres-solver/install/lib/cmake/Ceres
make -j8
```

### Run
```
export LD_LIBRARY_PATH=/orion/u/jingweih/3rd/cgal-install/lib:/orion/u/jingweih/3rd/ceres-solver/install/lib
./deform ./source.obj ./reference.obj output.obj 64 5000 10
```
10 is the rigidity strength

