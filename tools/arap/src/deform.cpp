#include <iostream>
#include <fstream>
#include <strstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/ceres.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/copyleft/marching_cubes.h>
#include <unordered_map>

typedef double FT;
typedef Eigen::Matrix<FT, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
typedef Eigen::Matrix<FT, Eigen::Dynamic, 1> VectorX;
typedef Eigen::SparseMatrix<FT> SpMat;
typedef Eigen::Triplet<FT> T;
typedef Eigen::Matrix<FT, 3, 1> Vector3;

int GRID_RESOLUTION = 64;
int MESH_RESOLUTION = 5000;

class UniformGrid
{
public:
	UniformGrid()
	: N(0)
	{}
	UniformGrid(int _N) {
		N = _N;
		distances.resize(N);
		for (auto& d : distances) {
			d.resize(N);
			for (auto& v : d)
				v.resize(N, 1e30);
		}
	}
	template <class T>
	T distance(const T* const p) const {
		int px = *(double*)&p[0] * N;
		int py = *(double*)&p[1] * N;
		int pz = *(double*)&p[2] * N;
		if (px < 0 || py < 0 || pz < 0 || px >= N - 1 || py >= N - 1 || pz >= N - 1) {
			T l = (T)0;
			if (px < 0)
				l = l + -p[0] * (T)N;
			else if (px >= N)
				l = l + (p[0] * (T)N - (T)(N - 1 - 1e-3));

			if (py < 0)
				l = l + -p[1] * (T)N;
			else if (py >= N)
				l = l + (p[1] * (T)N - (T)(N - 1 - 1e-3));

			if (pz < 0)
				l = l + -p[2] * (T)N;
			else if (pz >= N)
				l = l + (p[2] * (T)N - (T)(N - 1 - 1e-3));

			return l;
		}
		T wx = p[0] * (T)N - (T)px;
		T wy = p[1] * (T)N - (T)py;
		T wz = p[2] * (T)N - (T)pz;
		T w0 = ((T)1 - wx) * ((T)1 - wy) * ((T)1 - wz) * (T)distances[pz    ][py    ][px    ];
		T w1 = wx 		   * ((T)1 - wy) * ((T)1 - wz) * (T)distances[pz    ][py    ][px + 1];
		T w2 = ((T)1 - wx) * wy 		 * ((T)1 - wz) * (T)distances[pz    ][py + 1][px    ];
		T w3 = wx 		   * wy 		 * ((T)1 - wz) * (T)distances[pz    ][py + 1][px + 1];
		T w4 = ((T)1 - wx) * ((T)1 - wy) * wz 		   * (T)distances[pz + 1][py    ][px    ];
		T w5 = wx 		   * ((T)1 - wy) * wz 		   * (T)distances[pz + 1][py    ][px + 1];
		T w6 = ((T)1 - wx) * wy 		 * wz		   * (T)distances[pz + 1][py + 1][px    ];
		T w7 = wx 		   * wy 		 * wz 		   * (T)distances[pz + 1][py + 1][px + 1];
		T res = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7;

		return res;
	}
	int N;
	std::vector<std::vector<std::vector<FT> > > distances;
};

struct LengthError {
  LengthError(const Vector3& v_, FT lambda_)
  : v(v_), lambda(lambda_) {}

  template <typename T>
  bool operator()(const T* const p1,
                  const T* const p2,
                  T* residuals) const {
  	T px = p1[0] - p2[0];
  	T py = p1[1] - p2[1];
  	T pz = p1[2] - p2[2];
  	residuals[0] = px - (T)v[0];
  	residuals[1] = py - (T)v[1];
  	residuals[2] = pz - (T)v[2];
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const Vector3& v, const FT lambda_) {
     return (new ceres::AutoDiffCostFunction<LengthError, 3, 3, 3>(
                 new LengthError(v, lambda_)));
   }
   FT lambda;
   Vector3 v;
};

struct DistanceError {
  DistanceError(UniformGrid* grid_)
  : grid(grid_) {}

  template <typename T>
  bool operator()(const T* const p1,
                  T* residuals) const {
  	residuals[0] = grid->distance(p1);
  	residuals[1] = (T)0;
  	residuals[2] = (T)0;
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(UniformGrid* grid) {
     return (new ceres::AutoDiffCostFunction<DistanceError, 3, 3>(
                 new DistanceError(grid)));
   }
   UniformGrid* grid;
};

class Mesh
{
public:
	Mesh() : scale(1.0) {}
	std::vector<Vector3> V;
	std::vector<Eigen::Vector3i> F;

	void ReadOBJ(const char* filename) {
		std::ifstream is(filename);
		char buffer[256];
		while (is.getline(buffer, 256)) {
			std::strstream str;
			str << buffer;
			str >> buffer;
			if (strcmp(buffer, "v") == 0) {
				FT x, y, z;
				str >> x >> y >> z;
				V.push_back(Vector3(x, y, z));
			}
			else if (strcmp(buffer, "f") == 0) {
				Eigen::Vector3i f;
				for (int j = 0; j < 3; ++j) {
					str >> buffer;
					int id = 0;
					int p = 0;
					while (buffer[p] != '/') {
						id = id * 10 + (buffer[p] - '0');
						p += 1;
					}
					f[j] = id - 1;
				}
				F.push_back(f);
			}
		}
	}

	void ReadOBJ_Manifold(const char* filename) {
		std::ifstream is(filename);
		char buffer[256];
		while (is.getline(buffer, 256)) {
			std::strstream str;
			str << buffer;
			str >> buffer;
			if (strcmp(buffer, "v") == 0) {
				FT x, y, z;
				str >> x >> y >> z;
				V.push_back(Vector3(x, y, z));
			}
			else if (strcmp(buffer, "f") == 0) {
				Eigen::Vector3i f;
				int idx_x, idx_y, idx_z;
				str >> idx_x >> idx_y >> idx_z;
				// std::cout<<idx_x <<' '<< idx_y <<' '<< idx_z <<std::endl;
				f = Eigen::Vector3i(idx_x-1, idx_y-1, idx_z-1);
				F.push_back(f);
			}
		}
	}

	void WriteOBJ(const char* filename) {
		std::ofstream os(filename);
		for (int i = 0; i < V.size(); ++i) {
			os << "v " << V[i][0] << " " << V[i][1] << " " << V[i][2] << "\n";
		}
		for (int i = 0; i < F.size(); ++i) {
			os << "f " << F[i][0] + 1 << " " << F[i][1] + 1 << " " << F[i][2] + 1 << "\n";
		}
		os.close();
	}
	FT scale;
	Vector3 pos;
	void Normalize() {
		FT min_p[3], max_p[3];
		for (int j = 0; j < 3; ++j) {
			min_p[j] = 1e30;
			max_p[j] = -1e30;
			for (int i = 0; i < V.size(); ++i) {
				if (V[i][j] < min_p[j])
					min_p[j] = V[i][j];
				if (V[i][j] > max_p[j])
					max_p[j] = V[i][j];
			}
		}
		scale = std::max(max_p[0] - min_p[0], std::max(max_p[1] - min_p[1], max_p[2] - min_p[2])) * 1.1;
		for (int j = 0; j < 3; ++j)
			pos[j] = min_p[j] - 0.05 * scale;
		for (auto& v : V) {
			v = (v - pos) / scale;
		}
		for (int j = 0; j < 3; ++j) {
			min_p[j] = 1e30;
			max_p[j] = -1e30;
			for (int i = 0; i < V.size(); ++i) {
				if (V[i][j] < min_p[j])
					min_p[j] = V[i][j];
				if (V[i][j] > max_p[j])
					max_p[j] = V[i][j];
			}
		}
	}
	void ApplyTransform(Mesh& m) {
		pos = m.pos;
		scale = m.scale;
		for (auto& v : V) {
			v = (v - pos) / scale;
		}
	}
	void ConstructDistanceField(UniformGrid& grid) {
		MatrixX P(grid.N * grid.N * grid.N, 3);
		int offset = 0;
		for (int i = 0; i < grid.N; ++i) {
			for (int j = 0; j < grid.N; ++j) {
				for (int k = 0; k < grid.N; ++k) {
					P.row(offset) = Vector3(FT(k) / grid.N, FT(j) / grid.N, FT(i) / grid.N);
					offset += 1;
				}
			}
		}

		MatrixX V2(V.size(), 3);
		for (int i = 0; i < V.size(); ++i)
			V2.row(i) = V[i];

		Eigen::MatrixXi F2(F.size(), 3);
		for (int i = 0; i < F.size(); ++i)
			F2.row(i) = F[i];

		MatrixX N(F.size(), 3);
		for (int i = 0; i < F.size(); ++i) {
			Vector3 x = V[F[i][1]] - V[F[i][0]];
			Vector3 y = V[F[i][2]] - V[F[i][0]];
			N.row(i) = x.cross(y).normalized();
		}

		VectorX sqrD;
		Eigen::VectorXi I;
		MatrixX C;
		igl::point_mesh_squared_distance(P,V2,F2,sqrD,I,C);

		offset = 0;
		for (int i = 0; i < grid.N; ++i) {
			for (int j = 0; j < grid.N; ++j) {
				for (int k = 0; k < grid.N; ++k) {
					grid.distances[i][j][k] = sqrt(sqrD[offset]);
					offset += 1;
				}
			}
		}

	}

	void FromDistanceField(UniformGrid& grid) {
		VectorX S(grid.N * grid.N * grid.N);
		MatrixX GV(grid.N * grid.N * grid.N, 3);
		int offset = 0;
		for (int i = 0; i < grid.N; ++i) {
			for (int j = 0; j < grid.N; ++j) {
				for (int k = 0; k < grid.N; ++k) {
					S[offset] = grid.distances[i][j][k];
					GV.row(offset) = Vector3(k, j, i);
					offset += 1;
				}
			}
		}
		MatrixX SV;
		Eigen::MatrixXi SF;
		igl::copyleft::marching_cubes(S,GV,grid.N,grid.N,grid.N,SV,SF);
		V.resize(SV.rows());
		F.resize(SF.rows());
		for (int i = 0; i < SV.rows(); ++i)
			V[i] = SV.row(i) / (FT)grid.N;
		for (int i = 0; i < SF.rows(); ++i)
			F[i] = SF.row(i);
	}

	void DeformWithEdge(UniformGrid& grid, std::vector<Vector3>& V, std::vector<Vector3>& OV, std::vector<std::pair<int, int> > edges) {
		FT lambda = 2e-3;
		ceres::Problem problem;

		std::vector<double> V_data(V.size() * 3);
		double* pt = V_data.data();
		for (int i = 0; i < V.size(); ++i) {
			*pt++ = V[i][0];
			*pt++ = V[i][1];
			*pt++ = V[i][2];
		}
		//Move vertices
		std::vector<ceres::ResidualBlockId> v_block_ids;
		v_block_ids.reserve(V.size());
		for (int i = 0; i < V.size(); ++i) {
			ceres::CostFunction* cost_function = DistanceError::Create(&grid);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V_data.data() + i * 3);
			v_block_ids.push_back(block_id);			
		}

		//Enforce rigidity
		std::vector<ceres::ResidualBlockId> edge_block_ids;
		edge_block_ids.reserve(edges.size());
		for (auto& e : edges) {
			Vector3 v = (OV[e.first] - OV[e.second]);
			ceres::CostFunction* cost_function = LengthError::Create(v, lambda);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0,
				V_data.data() + e.first * 3,
				V_data.data() + e.second * 3);
			edge_block_ids.push_back(block_id);
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = false;
		options.num_threads = 1;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		//std::cout << summary.FullReport() << "\n";

		//V error
		ceres::Problem::EvaluateOptions v_options;
		v_options.residual_blocks = v_block_ids;
		double v_cost;
		problem.Evaluate(v_options, &v_cost, NULL, NULL, NULL);
		std::cout<<"Vertices cost: "<<v_cost<<std::endl;

		//E error
		ceres::Problem::EvaluateOptions edge_options;
		edge_options.residual_blocks = edge_block_ids;
		double edge_cost;
		problem.Evaluate(edge_options, &edge_cost, NULL, NULL, NULL);
		std::cout<<"Rigidity cost: "<<edge_cost<<std::endl;

		double final_cost = v_cost + edge_cost;
		std::cout<<"Final cost: "<<final_cost<<std::endl;

		pt = V_data.data();
		for (int i = 0; i < V.size(); ++i) {
			V[i][0] = *pt++;
			V[i][1] = *pt++;
			V[i][2] = *pt++;
		}
	}

	void HierarchicalDeform(UniformGrid& grid, std::vector<Vector3>& V, std::vector<std::pair<int, int> >& edges) {
		if (V.size() < MESH_RESOLUTION) {
			DeformWithEdge(grid, V, V, edges);
		} else {
			//Downsample
			std::vector<int> parents(V.size(), -1);
			std::vector<Vector3> lowres_V;
			std::vector<std::pair<int, int> > lowres_E;
			std::vector<std::pair<int, int> > collapsed_edges;
			std::vector<int> separate_points;
			for (int i = 0; i < edges.size(); ++i) {
				int v0 = edges[i].first;
				int v1 = edges[i].second;
				if (parents[v0] == -1 && parents[v1] == -1) {
					parents[v0] = lowres_V.size();
					parents[v1] = lowres_V.size();
					lowres_V.push_back(0.5 * (V[v0] + V[v1]));
					collapsed_edges.push_back(edges[i]);
				}
			}
			for (int i = 0; i < parents.size(); ++i) {
				if (parents[i] == -1) {
					parents[i] = lowres_V.size();
					lowres_V.push_back(V[i]);
					separate_points.push_back(i);
				}
			}
			std::unordered_set<long long> edges_key;
			long long kn = lowres_V.size();

			for (int i = 0; i < edges.size(); ++i) {
				int v0 = parents[edges[i].first];
				int v1 = parents[edges[i].second];
				if (v0 != v1) {
					if (v0 > v1)
						std::swap(v0, v1);
					long long key = (long long)v0 * (long long)kn + (long long) v1;
					if (!edges_key.count(key)) {
						lowres_E.push_back(std::make_pair(v0, v1));
						edges_key.insert(key);
					}
				}
			}
			HierarchicalDeform(grid, lowres_V, lowres_E);

			//Linear Estimation
			std::unordered_map<long long, FT> trips;
			int num_entries = V.size();
			MatrixX B = MatrixX::Zero(num_entries, 3);
			auto add_entry_A = [&](int x, int y, FT w) {
				long long key = (long long)x * (long long)num_entries + (long long)y;
				auto it = trips.find(key);
				if (it == trips.end())
					trips[key] = w;
				else
					it->second += w;
			};
			auto add_entry_B = [&](int m, const Vector3& v) {
				B.row(m) += v;
			};

			for (auto& e : collapsed_edges) {
				int v0 = e.first;
				int v1 = e.second;
				int vid = parents[v0];
				add_entry_A(v0, v0, 0.25);
				add_entry_A(v0, v1, 0.25);
				add_entry_A(v1, v0, 0.25);
				add_entry_A(v1, v1, 0.25);
				add_entry_B(v0, 0.5 * lowres_V[vid]);
				add_entry_B(v1, 0.5 * lowres_V[vid]);
			}

			for (auto& v : separate_points) {
				int vid = parents[v];
				add_entry_A(v, v, 1);
				add_entry_B(v, lowres_V[vid]);
			}
			FT regular = 2e-3;
			for (auto& e : edges) {
				int v0 = e.first;
				int v1 = e.second;
				add_entry_A(v0, v0, regular);
				add_entry_A(v0, v1, -regular);
				add_entry_A(v1, v0, -regular);
				add_entry_A(v1, v1, regular);
				add_entry_B(v0, regular * (V[v0] - V[v1]));
				add_entry_B(v1, regular * (V[v1] - V[v0]));
			}
			Eigen::SparseMatrix<FT> A(num_entries, num_entries);
			std::vector<T> tripletList;
			for (auto& m : trips) {
				tripletList.push_back(T(m.first / num_entries, m.first % num_entries, m.second));
			}
			A.setFromTriplets(tripletList.begin(), tripletList.end());

			Eigen::SparseLU<Eigen::SparseMatrix<FT>> solver;
	        solver.analyzePattern(A);

	        solver.factorize(A);

	        //std::vector<Vector3> NV(V.size());
	        for (int j = 0; j < 3; ++j) {
		        VectorX result = solver.solve(B.col(j));

		        for (int i = 0; i < result.rows(); ++i) {
		        	V[i][j] = result[i];
		        }
		    }

		    //DeformWithEdge(grid, NV, V, edges);
		    //V = NV;
		}
	}

	void HierarchicalDeform(UniformGrid& grid) {
		std::vector<std::pair<int, int> > edges;
		for (int i = 0; i < F.size(); ++i) {
			for (int j = 0; j < 3; ++j) {
				if (F[i][j] < F[i][(j + 1) % 3])
					edges.push_back(std::make_pair(F[i][j], F[i][(j + 1) % 3]));
			}
		}
		HierarchicalDeform(grid, V, edges);
	}

	void Deform(UniformGrid& grid) {
		FT lambda = 1e-3;
		ceres::Problem problem;

		//Move vertices
		std::vector<ceres::ResidualBlockId> v_block_ids;
		v_block_ids.reserve(V.size());
		for (int i = 0; i < V.size(); ++i) {
			ceres::CostFunction* cost_function = DistanceError::Create(&grid);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[i].data());
			v_block_ids.push_back(block_id);			
		}

		//Enforce rigidity
		std::vector<ceres::ResidualBlockId> edge_block_ids;
		edge_block_ids.reserve(3 * F.size());
		for (int i = 0; i < F.size(); ++i) {
			for (int j = 0; j < 3; ++j) {
				Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
				ceres::CostFunction* cost_function = LengthError::Create(v, lambda);
				ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[F[i][j]].data(), V[F[i][(j + 1) % 3]].data());
				edge_block_ids.push_back(block_id);
			}
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = true;
		options.num_threads = 1;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << "\n";

		//V error
		ceres::Problem::EvaluateOptions v_options;
		v_options.residual_blocks = v_block_ids;
		double v_cost;
		problem.Evaluate(v_options, &v_cost, NULL, NULL, NULL);
		std::cout<<"Vertices cost: "<<v_cost<<std::endl;

		//E error
		ceres::Problem::EvaluateOptions edge_options;
		edge_options.residual_blocks = edge_block_ids;
		FT edge_cost;
		problem.Evaluate(edge_options, &edge_cost, NULL, NULL, NULL);
		std::cout<<"Rigidity cost: "<<edge_cost<<std::endl;

		FT final_cost = v_cost + edge_cost;
		std::cout<<"Final cost: "<<final_cost<<std::endl;
	}

	FT Get_Final_Cost(Mesh& ref){

		//normalize by number of vertices and faces
		int ref_num_v = ref.V.size();
		int source_num_f = F.size();

		MatrixX SV(V.size(), 3), RV(ref.V.size(), 3);
		Eigen::MatrixXi SF(F.size(), 3);
		for (int i = 0; i < V.size(); ++i)
			SV.row(i) = V[i];
		for (int i = 0; i < ref.V.size(); ++i)
			RV.row(i) = ref.V[i];
		for (int i = 0; i < F.size(); ++i)
			SF.row(i) = F[i];


		VectorX sqrD;
		Eigen::VectorXi I;
		MatrixX C;
		igl::point_mesh_squared_distance(RV, SV, SF,sqrD,I,C);
		FT coverage_cost = sqrD.sum() * 0.5;
		std::cout<<"Coverage cost: "<<coverage_cost << std::endl;

		FT rigidity_cost = 0.0;
		std::cout<<"Rigidity cost: "<<rigidity_cost<<std::endl;

		FT final_cost = coverage_cost/ref_num_v + rigidity_cost/source_num_f;
		std::cout<<"Final cost: "<<final_cost<<std::endl;

		return final_cost;
	}

};

int main(int argc, char** argv) {	
	if (argc < 5) {
		printf("./deform source.obj reference.obj output.obj textfile_name [GRID_RESOLUTION=64] [MESH_RESOLUTION=5000]\n");
		return 0;
	}
	//Deform source to fit the reference

	Mesh src, ref;
	src.ReadOBJ_Manifold(argv[1]);
	ref.ReadOBJ_Manifold(argv[2]);

	if (argc > 5)
		sscanf(argv[5], "%d", &GRID_RESOLUTION);

	if (argc > 6)
		sscanf(argv[6], "%d", &MESH_RESOLUTION);

	//Get number of vertices and faces
	std::cout<<"Source:\t\t"<<"Num vertices: "<<src.V.size()<<"\tNum faces: "<<src.F.size()<<std::endl;
	std::cout<<"Reference:\t"<<"Num vertices: "<<ref.V.size()<<"\tNum faces: "<<ref.F.size()<<std::endl<<std::endl;

	UniformGrid grid(GRID_RESOLUTION);
	ref.Normalize();
	src.ApplyTransform(ref);

	ref.ConstructDistanceField(grid);
	src.HierarchicalDeform(grid);

	FT cost;
	cost = src.Get_Final_Cost(ref);

	FT threshold = 1e-4;

	//for deform_tune
	// FT threshold = 1e-3;
	// FT threshold = 5e-4; #does not filter valid deformations

	if (cost > threshold){
		std::cout<<"INVALID"<<std::endl;
		return 0;
	}

	std::cout<<"Deformed"<<std::endl;
	std::string text_file_name = argv[4];
	std::ofstream outfile;
	outfile.open(text_file_name, std::fstream::in | std::fstream::out | std::fstream::app);
	if (!outfile){
		outfile.open(text_file_name,  std::fstream::in | std::fstream::out | std::fstream::trunc);
	} 
	//output_obj\tcost
	outfile<<argv[3]<<'\t'<<cost<<std::endl;
	outfile.close();

	std::ifstream is(argv[1]);
	std::ofstream os(argv[3]);
	char buffer[1024];
	int offset = 0;
	while (is.getline(buffer, 1024)) {
		if (buffer[0] == 'v' && buffer[1] == ' ') {
			auto v = src.V[offset++] * src.scale + src.pos;
			os << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
		} else {
			os << buffer << "\n";
		}
	}
	is.close();
	os.close();
	return 0;
}
