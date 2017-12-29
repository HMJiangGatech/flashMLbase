//
// Created by haoming on 2017/10/10.
// Description: 
//
#include <fmlbase/PISTALassoSolver.h>

namespace fmlbase{
    fmlbase::PISTALassoSolver::PISTALassoSolver(const fmlbase::utils::FmlParam &param) : PIS2TASQRTLassoSolver(param) {
        // reset lambda
        VectorXd grad0(nfeature);
        loss_grad(grad0);
        lambdas[0] = grad0.cwiseAbs().maxCoeff();
        lambda = lambdas[0];
        sigma = param.getDoubleArg("sigma");
        double min_lambda_ratio;
        if(param.hasArg("minlambda_ratio"))
            min_lambda_ratio = sigma*param.getDoubleArg("minlambda_ratio");
        else
            min_lambda_ratio = sigma*sqrt(log(nfeature)/ntrain_sample) / lambdas[0];
        double anneal_lambda = pow(min_lambda_ratio,1./(niter-1));
        for (int i = 1; i < niter; ++i) {
            lambdas[i] = lambdas[i-1]*anneal_lambda;
        }

        // setting epsilon
        if(param.hasArg("epsilon"))
            epsilon = param.getDoubleArg("epsilon");
        else
            epsilon = lambdas[niter-1] * 0.5;

        hessianMat = nullptr;
        stepsize_max = this->hessian().norm()*param.getDoubleArg("stepsize_scale");
    }
    void PISTALassoSolver::reinitialize() {
        PIS2TASQRTLassoSolver::reinitialize();
    }

}// namespace fmlbase
