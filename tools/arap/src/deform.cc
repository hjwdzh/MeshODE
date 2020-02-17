#include "deform.h"

#include <ceres/ceres.h>

#include "distanceloss.h"
#include "edgeloss.h"

void Deform(Mesh& mesh, UniformGrid& grid, FT lambda) {
	auto& V = mesh.V;
	auto& F = mesh.F;
	
	ceres::Problem problem;

	//Move vertices
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(3 * F.size());
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
			ceres::CostFunction* cost_function = EdgeLoss::Create(v, lambda);
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

void DeformWithRot(Mesh& mesh, UniformGrid& grid, FT lambda) {
	auto& V = mesh.V;
	auto& F = mesh.F;
	
	ceres::Problem problem;

	//Move vertices
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(3 * F.size());
	std::vector<double> rots(V.size() * 3, 0);
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
			ceres::CostFunction* cost_function = EdgeLossWithRot::Create(v, lambda);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0,
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
