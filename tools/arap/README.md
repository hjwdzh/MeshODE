# ARAP

### Build
```
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Run
```
./deform ./source.obj ./reference.obj ./cad.obj output.obj 64 5000 10
```
10 is the rigidity strength

