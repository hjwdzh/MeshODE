#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cad_layer.h"
#include "distance_layer.h"
#include "graph_layer.h"
#include "rigid_layer.h"

namespace py = pybind11;
//#define USE_DOUBLE

PYBIND11_MODULE(pyDeform, m) {
	m.def("LoadMesh", &LoadMesh);
	m.def("LoadCadMesh", &LoadCadMesh);
	m.def("SaveMesh", &SaveMesh);

	m.def("InitializeDeformTemplate", &InitializeDeformTemplate);
	m.def("NormalizeByTemplate", &NormalizeByTemplate);
	m.def("DenormalizeByTemplate", &DenormalizeByTemplate);

	m.def("DistanceFieldLoss_forward", &DistanceFieldLoss_forward);
	m.def("DistanceFieldLoss_backward", &DistanceFieldLoss_backward);

	m.def("RigidEdgeLoss_forward", &RigidEdgeLoss_forward);
	m.def("RigidEdgeLoss_backward", &RigidEdgeLoss_backward);
	m.def("StoreRigidityInformation", &StoreRigidityInformation);

	m.def("CadEdgeLoss_forward", &CadEdgeLoss_forward);
	m.def("CadEdgeLoss_backward", &CadEdgeLoss_backward);
	m.def("StoreCadInformation", &StoreCadInformation);

	m.def("GraphEdgeLoss_forward", &GraphEdgeLoss_forward);
	m.def("GraphEdgeLoss_backward", &GraphEdgeLoss_backward);
	m.def("StoreGraphInformation", &StoreGraphInformation);

}

