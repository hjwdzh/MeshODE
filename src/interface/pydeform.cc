#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "rigid_layer.h"

namespace py = pybind11;
//#define USE_DOUBLE

PYBIND11_MODULE(pyDeform, m) {
	m.def("LoadMesh", &LoadMesh);
	m.def("SaveMesh", &SaveMesh);
	m.def("InitializeDeformTemplate", &InitializeDeformTemplate);
	m.def("NormalizeByTemplate", &NormalizeByTemplate);
	m.def("DenormalizeByTemplate", &DenormalizeByTemplate);
	m.def("DistanceFieldLoss_forward", &DistanceFieldLoss_forward);
	m.def("DistanceFieldLoss_backward", &DistanceFieldLoss_backward);
	m.def("EdgeLoss_forward", &EdgeLoss_forward);
	m.def("EdgeLoss_backward", &EdgeLoss_backward);
	m.def("StoreRigidityInformation", &StoreRigidityInformation);
}

