# Deep Shape Deformation.
Deform Shape A to fit shape B.

![Plane Fitting Results](https://github.com/hjwdzh/ShapeDeform/raw/master/res/teaser.png)

### Dependencies
1. libIGL
2. CGAL
3. Ceres
4. pytorch

### Installing prerequisites
```
# recursively clone all 3rd party submodules
bash get_submodules.sh

# install python requirements
pip install -r requirements.txt

# install CERES (For Ubuntu)
sudo apt-get install cmake
sudo apt-get install libgoogle-glog-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libsuitesparse-dev
sudo add-apt-repository ppa:bzindovic/suitesparse-bugfix-1319687
sudo apt-get update
sudo apt-get install libsuitesparse-dev
mkdir 3rd_party/ceres-solver/ceres-bin
cd 3rd_party/ceres-solver/ceres-bin
cmake -DEXPORT_BUILD_DIR=ON ..
make -j4
make test
sudo make install

# install CERES (for Mac, Homebrew)
brew install ceres-solver --HEAD
```

### Build
```
mkdir build
cmake ..
make -j8
```
Note that when torch must be imported before the pyDeform package in Python. i.e.,
```python
import torch
import pyDeform
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
python ../src/python/cad_deform2.py ../data/cad.obj ../data/reference.obj ./cad_output.obj 1
python ../src/python/cad_neural_deform2.py ../data/cad.obj ../data/reference.obj ./cad_output.obj 0 cuda
```

