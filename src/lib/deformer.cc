#include "deformer.h"

#include <ceres/ceres.h>
#include <igl/point_mesh_squared_distance.h>

#include "distanceloss.h"
#include "edgeloss.h"

Deformer::Deformer(FT lambda, CallBackFunc func)
: lambda_(lambda), callback_(0)
{
	if (func != 0)
		callback_ = std::shared_ptr<TerminateWhenSuccessCallback>(
			new TerminateWhenSuccessCallback(func));
}


void Deformer::Deform(const UniformGrid& grid, Mesh* pmesh) {

	Mesh& mesh = *pmesh;
	FT lambda = lambda_;
	TerminateWhenSuccessCallback* callback =
		callback_ == 0 ? 0 : &(*callback_);
	auto& V = mesh.GetV();
	auto& F = mesh.GetF();
	
	ceres::Problem problem;

	//Move vertices
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(
			cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(3 * F.size());
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
			ceres::CostFunction* cost_function = EdgeLoss::Create(v, lambda);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(
				cost_function, 0,
				V[F[i][j]].data(),
				V[F[i][(j + 1) % 3]].data()
			);
			edge_block_ids.push_back(block_id);
		}
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 1;
	if (callback) {
		double prev_cost = 1e30;
		options.callbacks.push_back(callback);

		while (true) {
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			if (std::abs(prev_cost - summary.final_cost) < 1e-6)
				break;
			prev_cost = summary.final_cost;
		}
	} else {
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	}

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

void Deformer::DeformWithRot(const UniformGrid& grid, Mesh* pmesh) {

	Mesh& mesh = *pmesh;
	FT lambda = lambda_;
	TerminateWhenSuccessCallback* callback =
		callback_ == 0 ? 0 : &(*callback_);
	auto& V = mesh.GetV();
	auto& F = mesh.GetF();
	
	ceres::Problem problem;

	//Move vertices
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id =
			problem.AddResidualBlock(cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(3 * F.size());
	std::vector<double> rots(V.size() * 3, 0);
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
			ceres::CostFunction* cost_function =
				EdgeLossWithRot::Create(v, lambda);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(
				cost_function, 0,
				V[F[i][j]].data(),
				V[F[i][(j + 1) % 3]].data(),
				rots.data() + F[i][j] * 3,
				rots.data() + F[i][(j + 1) % 3] * 3
				);
			edge_block_ids.push_back(block_id);
		}
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 1;
	if (callback) {
		double prev_cost = 1e30;
		options.callbacks.push_back(callback);

		while (true) {
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			if (std::abs(prev_cost - summary.final_cost) < 1e-6)
				break;
			prev_cost = summary.final_cost;
		}
	} else {
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	}
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

void Deformer::DeformSubdivision(const UniformGrid& grid, Subdivision* psub) {

	Subdivision& sub = *psub;
	FT lambda = lambda_;
	TerminateWhenSuccessCallback* callback =
		callback_ == 0 ? 0 : &(*callback_);
	auto& mesh = sub.GetMesh();
	auto& V = mesh.GetV();
	auto& F = mesh.GetF();
	
	ceres::Problem problem;

	//Move vertices 
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(
			cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(3 * F.size() + sub.Neighbors().size());

	for (auto& p : sub.Neighbors()) {
		int v1 = p.first;
		int v2 = p.second;
		Vector3 v = (V[v1] - V[v2]);
		ceres::CostFunction* cost_function =
			AdaptiveEdgeLoss::Create(v, lambda);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(
			cost_function, 0, V[v1].data(), V[v2].data());
		edge_block_ids.push_back(block_id);
	}

	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
			
			ceres::CostFunction* cost_function =
				AdaptiveEdgeLoss::Create(v, lambda);
			
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(
				cost_function, 0,
				V[F[i][j]].data(),
				V[F[i][(j + 1) % 3]].data());

			edge_block_ids.push_back(block_id);
		}
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 1;
	if (callback) {
		double prev_cost = 1e30;
		options.callbacks.push_back(callback);

		while (true) {
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			if (std::abs(prev_cost - summary.final_cost) < 1e-6)
				break;
			prev_cost = summary.final_cost;
		}
	} else {
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	}

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

void Deformer::ReverseDeform(const Mesh& tar, Mesh* psrc) {
	Mesh& src = *psrc;
	FT lambda = lambda_;

	typedef Eigen::Matrix<FT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
		MatrixRowMajor;

	auto& src_V = src.GetV();
	auto& src_F = src.GetF();
	auto& tar_V = tar.GetV();
	auto& tar_F = tar.GetF();

	MatrixRowMajor V1(src_V.size(), 3), V2(tar_V.size(), 3);
	Eigen::MatrixXi	F1(src_F.size(), 3), F2(tar_F.size(), 3);

	for (int i = 0; i < src_V.size(); ++i)
		V1.row(i) = src_V[i];

	for (int i = 0; i < tar_V.size(); ++i)
		V2.row(i) = tar_V[i];

	for (int i = 0; i < src_F.size(); ++i)
		F1.row(i) = src_F[i];

	for (int i = 0; i < tar_F.size(); ++i)
		F2.row(i) = tar_F[i];

	TerminateWhenSuccessCallback callback;

	double prev_cost = 1e30;
	int step = 0;
	std::vector<ceres::CostFunction*> cost_function1, cost_function2;
	auto Vc = V1;
	while (true) {
		MatrixX sqrD;
		Eigen::VectorXi I;
		MatrixX C;

		igl::point_mesh_squared_distance(V2,V1,F1,sqrD,I,C);

		ceres::Problem problem;

		auto& V = V1;
		auto& F = F1;

		//Move vertices

		for (int i = 0; i < C.rows(); ++i) {
			int find = I[i];
			MatrixX weight;
			igl::barycentric_coordinates(C.row(i),
				V1.row(F1(find, 0)),
				V1.row(F1(find, 1)),
				V1.row(F1(find, 2)),
				weight);

			Vector3 w = weight.row(0);
			ceres::CostFunction* cost_function =
				BarycentricDistanceLoss::Create(w, V2.row(i));

			problem.AddResidualBlock(cost_function, 0,
				&V1(F1(find, 0), 0),&V1(F1(find, 1), 0),&V1(F1(find, 2), 0));
		}

		//Enforce rigidity
		for (int i = 0; i < V.rows(); ++i) {
			ceres::CostFunction* cost_function =
				PointRegularizerLoss::Create(1e-3, Vc.row(i));
			problem.AddResidualBlock(cost_function, 0, &(V(i,0)));
		}

		for (int i = 0; i < F.rows(); ++i) {
			for (int j = 0; j < 3; ++j) {
				Vector3 v = (Vc.row(F(i,j)) - Vc.row(F(i,(j + 1) % 3)));
				ceres::CostFunction* cost_function =
					EdgeLoss::Create(v, lambda);
				problem.AddResidualBlock(cost_function, 0,
					&V(F(i,j),0),
					&V(F(i,(j + 1) % 3),0));
			}
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = false;
		options.num_threads = 1;

		options.callbacks.push_back(&callback);

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (std::abs(summary.final_cost - prev_cost) < 1e-6)
			break;
		prev_cost = summary.final_cost;
		step += 1;
		if (step == 30)
			break;
	}

	for (int i = 0; i < src_V.size(); ++i)
		src_V[i] = V1.row(i);
}

void Deformer::DeformGraph(const UniformGrid& grid, Subdivision* sub) {
	auto& V = sub->GraphV();
	auto& E = sub->GraphE();

	FT lambda = lambda_;
	TerminateWhenSuccessCallback* callback =
		callback_ == 0 ? 0 : &(*callback_);

	
	ceres::Problem problem;

	//Move vertices
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(
			cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(E.size());
	for (auto& info : E) {
		Vector3 v = V[info.first] - V[info.second];
		ceres::CostFunction* cost_function = EdgeLoss::Create(v, lambda);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(
			cost_function, 0,
			V[info.first].data(),
			V[info.second].data()
		);
		edge_block_ids.push_back(block_id);		
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 1;
	if (callback) {
		double prev_cost = 1e30;
		options.callbacks.push_back(callback);

		while (true) {
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			if (std::abs(prev_cost - summary.final_cost) < 1e-6)
				break;
			prev_cost = summary.final_cost;
		}
	} else {
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	}

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