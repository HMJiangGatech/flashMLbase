//
// Created by haoming on 2017/10/10.
// Description: 
//
#include <fmlbase/PISTALassoSolver.h>

namespace fmlbase{
    PISTALassoSolver::PISTALassoSolver(const fmlbase::utils::FmlParam &param) : PIS2TASQRTLassoSolver(param) {
    }
    PISTALassoSolver::PISTALassoSolver(const utils::FmlParam &param, const MatrixXd &design_mat,
                                       const VectorXd &response_vec) : PIS2TASQRTLassoSolver(param, design_mat,
                                                                                             response_vec) {

    }
    void PISTALassoSolver::initialize() {

        theta = new VectorXd(nparameter);
        theta->setZero();
        thetas.emplace_back(theta);

        if(solver_param->hasArg("niter"))
            niter = solver_param->getIntArg("niter");
        else
            niter = 100;

        // initializing lambdas
        lambda = 0;
        lambdas = new double[niter];
        VectorXd grad0(nparameter);
        loss_grad(grad0);
        lambdas[0] = grad0.cwiseAbs().maxCoeff();
        lambda = lambdas[0];
        sigma = solver_param->getDoubleArg("sigma");
        double min_lambda_ratio;
        if(solver_param->hasArg("minlambda_ratio"))
            min_lambda_ratio = sigma*solver_param->getDoubleArg("minlambda_ratio");
        else
            min_lambda_ratio = sigma*sqrt(log(nfeature)/ntrain_sample) / lambdas[0];
        double anneal_lambda = pow(min_lambda_ratio,1./(niter-1));
        for (int i = 1; i < niter; ++i) {
            lambdas[i] = lambdas[i-1]*anneal_lambda;
        }

        // setting epsilon
        if(solver_param->hasArg("epsilon"))
            epsilon = solver_param->getDoubleArg("epsilon");
        else
            epsilon = lambdas[niter-1] * 0.5;

        hessianMat = nullptr;
        stepsize_max = this->hessian().norm()*solver_param->getDoubleArg("stepsize_scale");
    }
    void PISTALassoSolver::reinitialize() {
        PIS2TASQRTLassoSolver::reinitialize();
    }



}// namespace fmlbase
