# MeshODE: A Robust and Scalable Framework for Mesh Deformation
Pairwise shape deformation.

![Plane Fitting Results](https://github.com/hjwdzh/ShapeDeform/raw/master/res/teaser.jpg)

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

### Download data
We already provide some demo shapes in the data folder. For playing with more shapes, try
```
cd data
wget -i filelist.txt
```
You will get 3625 pairs of shapes.

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
./rigid_deform ../data/source.obj ../data/target.obj output.obj [GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] [lambda=1] [symmetry=0].
./cad_deform ../data/cad.obj ../data/target.obj output.obj [GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] [lambda=1] [symmetry=0].
```

### Run Pytorch optimizer
```
cd build
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
1. For watertight mesh deformation, try
```
python ../src/python/rigid_deform.py --source ../data/source.obj --target ../data/target.obj --output ./rigid_output.obj
```
2. For general CAD model deformation using traditional method
```
python ../src/python/cad_deform2.py --source ../data/cad-source.obj --target ../data/cad-target.obj --output ./cad_output.obj --rigidity 1
```
3. For NeuarlODE-based deformation, try
```
python ../src/python/cad_neural_deform2.py --source ../data/cad-source.obj --target ../data/cad-target.obj --output ./cad_output.obj --save_path ./cad_output.ckpt --rigidity 0.1 --device cpu [cuda if possible for faster optimization]
```
4. To generate intermediate steps during deformation with NeuralODE (assuming you have previous script done), try
```
python ../src/python/cad_neural_animate.py --source ../data/cad-source.obj --target ../data/cad-target.obj --output_folder ./animation --rigidity 0.1 --resume_path ./cad_output.ckpt --device cpu [cuda if possible for faster optimization]
```

## Author
- [Jingwei Huang](mailto:jingweih@stanford.edu)

&copy; 2020 Jingwei Huang All Rights Reserved

**IMPORTANT**: If you use this code please cite the following (to provide) in any resulting publication:
```
To appear.
```
