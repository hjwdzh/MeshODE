#ifndef SHAPEDEFORM_CALLBACK_H_
#define SHAPEDEFORM_CALLBACK_H_

#include <ceres/ceres.h>

typedef void (*CallBackFunc)(void);

class TerminateWhenSuccessCallback : public ceres::IterationCallback{
public:
	TerminateWhenSuccessCallback()
	{
		func_ = 0;
		counter_ = 0;
	}
	TerminateWhenSuccessCallback(CallBackFunc func)
	{
		func_ = func;
		counter_ = 0;
	}
	~TerminateWhenSuccessCallback() {}

	ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
	{
		if (summary.step_is_successful) {
			if (counter_ == 0) {
				if (func_)
					func_();
				counter_ = 1;
				return ceres::SOLVER_CONTINUE;
			}
			counter_ = 0;
			return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
		}

		return ceres::SOLVER_CONTINUE;
	}

private:
	CallBackFunc func_;
	int counter_;
};

#endif