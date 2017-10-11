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
            min_lambda_ratio = param.getIntArg("minlambda_ratio");
        else
            min_lambda_ratio = sigma*sqrt(log(nfeature)/ntrain_sample) / lambdas[0];
        double anneal_lambda = pow(min_lambda_ratio,1./niter);
        for (int i = 1; i < niter; ++i) {
            lambdas[i] = lambdas[i-1]*anneal_lambda;
        }
        hessianMat = nullptr;
        stepsize_max = this->hessian().norm()*100;
    }
}// namespace fmlbase
