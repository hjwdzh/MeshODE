#ifndef ARAP_CALLBACK_H_
#define ARAP_CALLBACK_H_

#include <ceres/ceres.h>

typedef void (*callback_function)(void);

class TerminateWhenSuccessCallback : public ceres::IterationCallback{
public:
	TerminateWhenSuccessCallback()
	{
		func = 0;
		counter = 0;
	}
	TerminateWhenSuccessCallback(callback_function func_)
	{
		func = func_;
		counter = 0;
	}
	~TerminateWhenSuccessCallback() {}
	ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
		if (summary.step_is_successful) {
			if (counter == 0) {
				if (func) func();
				counter = 1;
				return ceres::SOLVER_CONTINUE;
			}
			counter = 0;
			return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
		}

		return ceres::SOLVER_CONTINUE;
	}
	callback_function func;
	int counter;
};

#endif