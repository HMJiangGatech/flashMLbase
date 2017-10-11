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
        double anneal_lambda = pow(min_lambda_ratio,1./(niter-1));
        for (int i = 1; i < niter; ++i) {
            lambdas[i] = lambdas[i-1]*anneal_lambda;
        }

        // setting epsilon
        if(param.hasArg("epsilon"))
            epsilon = param.getDoubleArg("epsilon");
        else
            epsilon = lambdas[niter-1] * 0.25;

        hessianMat = nullptr;
        stepsize_max = this->hessian().norm();
    }
    void PISTALassoSolver::reinitialize() {
        PIS2TASQRTLassoSolver::reinitialize();
    }

    void PISTALassoSolver::train() {
        for (int i = 1; i < niter; ++i) {
            if(verbose)
                std::cout << "Outer Loop: " << i << std::endl;
            theta = new VectorXd(*thetas.back());
            lambda = lambdas[i];
            double k_epsilon;
            if(i < niter-1)
                k_epsilon = lambda;     //! MODIFIED:   lambda*0.25
            else
                k_epsilon = epsilon;
            double k_stepsize = stepsize_max;
            ISTA(k_stepsize, k_epsilon);
            thetas.emplace_back(theta);
            if(verbose)
            {
                std::cout   << "  error: "<< k_epsilon
                            << "  obj val: "<< obj_value()
                            << "  loss val: "<< loss_value()
                            << "  hessian norm" << this->hessian().norm()
                            <<std::endl;
            }
        }
    }

    void PISTALassoSolver::ISTA(double k_stepsize, double k_epsilon) {
        int t = 0;
        double maxkss = -1;
        while(++t != 0)
        {
            VectorXd grad;
            loss_grad(grad);
            double tau;

            // backtracking line search.
            double tilde_stepsize = k_stepsize;
            VectorXd temp_theta(nfeature);
            bool exitflag1 = false, exitflag2 = false;
            while (true){
                tau = lambda/tilde_stepsize;
                temp_theta = *theta - grad/tilde_stepsize;
                temp_theta = temp_theta.cwiseSign().cwiseProduct((temp_theta.array().abs() - tau).max(0).matrix());
                double q_val = this->q_value(temp_theta,grad,tilde_stepsize);
                //std::cout <<verbose<<"\t\tobj val :" << obj_value(&temp_theta) << " quadratic approximation: " << q_val<<std::endl;
                if (obj_value(&temp_theta) < q_val){
                    tilde_stepsize *= 0.5;
                    if(exitflag1)
                        exitflag2 = true;
                }  else
                {
                    exitflag1 = true;
                    tilde_stepsize *= 2;
                    if (tilde_stepsize>stepsize_max)
                        exitflag2 = true;
                }
                if(exitflag1 and exitflag2)
                    break;
            }
            // update theta
            k_stepsize = std::min(tilde_stepsize, stepsize_max);

            tau = lambda/k_stepsize;
            temp_theta = *theta - grad/k_stepsize;
            temp_theta = temp_theta.cwiseSign().cwiseProduct((temp_theta.array().abs() - tau).max(0).matrix());
            *theta = temp_theta;

            // check stopping criteria
            obj_grad(grad);
            grad = grad.cwiseAbs();
            grad = (grad.array() - (1 - theta->cwiseSign().cwiseAbs().array()) * lambda).matrix();
            double omega = grad.maxCoeff();
            if(verbose){
                std::cout << "  \tMiddle Loop: " << t
                          << "  AKKT:" << omega
                          << "  Desired Precision: " << k_epsilon
                          << "  k_stepsize: "<< k_stepsize
                          << "  obj val: "<< obj_value()
                          << "  loss val: "<< loss_value()
                          << std::endl;
                maxkss = std::max(maxkss,k_stepsize);
            }
            if(omega <= k_epsilon)
                break;
        }
        if(verbose)
            std::cout<<"  max stepsize: "<< maxkss<<std::endl;
    }

}// namespace fmlbase
