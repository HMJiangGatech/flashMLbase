// PIS2TA SQRT Lasso Solver (Pathwise Optimization Iterative Shrinkage Thresholding)
// Created by haoming on 2017/10/5.
// Description: 
//
#include <fmlbase/PIS2TASQRTLassoSolver.h>


namespace fmlbase{

    PIS2TASQRTLassoSolver::PIS2TASQRTLassoSolver(const utils::FmlParam &param) : SolverBase(param), nlambda(niter){

        theta = new VectorXd(nfeature);
        theta->setZero();
        thetas.emplace_back(theta);

        if(param.hasArg("niter"))
            niter = param.getIntArg("niter");
        else
            niter = 100;

        // initializing lambdas
        lambda = 0;
        lambdas = new double[niter];
        VectorXd grad0(nfeature);
        loss_grad(grad0);
        lambdas[0] = grad0.cwiseAbs().maxCoeff();
        lambda = lambdas[0];
        double min_lambda_ratio;
        if(param.hasArg("minlambda_ratio"))
            min_lambda_ratio = param.getIntArg("minlambda_ratio");
        else
            min_lambda_ratio = sqrt(log(nfeature)/ntrain_sample) / lambdas[0];   //! different from the paper
        double anneal_lambda = pow(min_lambda_ratio,1./(niter-1));
        for (int i = 1; i < niter; ++i) {
            lambdas[i] = lambdas[i-1]*anneal_lambda;
        }

        // setting epsilon
        if(param.hasArg("epsilon"))
            epsilon = param.getDoubleArg("epsilon");
        else
            epsilon = lambdas[niter-1] * 0.25;

        stepsize_max = this->hessian().norm();
    }

    PIS2TASQRTLassoSolver::~PIS2TASQRTLassoSolver() {
        for (auto &theta : thetas)
            delete theta;
        delete lambdas;
    }

    void PIS2TASQRTLassoSolver::reinitialize() {
        for (auto &theta : thetas)
            delete theta;
        thetas.clear();

        theta = new VectorXd(nfeature);
        theta->setZero();
        thetas.emplace_back(theta);
    }

    void PIS2TASQRTLassoSolver::train() {
        for (int i = 1; i < niter; ++i) {
            if(verbose)
                std::cout << "Outer Loop: " << i << std::endl;
            theta = new VectorXd(*thetas.back());
            lambda = lambdas[i];
            double k_epsilon;
            if(i < niter-1)
                k_epsilon = lambda*0.5;     //! MODIFIED:   lambda*0.25
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

    void PIS2TASQRTLassoSolver::ISTA(double k_stepsize, double k_epsilon) {
        int t = 0;
        double maxkss = -1;
        while(++t != 0)
        {
            VectorXd grad;
            loss_grad(grad);
            double tau;

            // backtracking line search.
            double tilde_stepsize;
            tilde_stepsize = k_stepsize;
            VectorXd temp_theta(nfeature);
            bool exitflag1 = false, exitflag2 = false;
            while (true){
                tau = lambda/tilde_stepsize;
                temp_theta = *theta - grad/tilde_stepsize;
                temp_theta = temp_theta.cwiseSign().cwiseProduct((temp_theta.array().abs() - tau).max(0).matrix());
                double q_val = this->q_value(temp_theta,grad,tilde_stepsize);
                //std::cout <<"\t\tobj val :" << obj_value(&temp_theta) << " quadratic approximation: " << q_val<<std::endl;
                if (obj_value(&temp_theta) < q_val)
                    tilde_stepsize *= 0.5;
                else
                    break;
            }
            // update theta
            k_stepsize = std::min(tilde_stepsize*2, stepsize_max);

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
                          << "  lambda: "<< lambda
                          << std::endl;
                maxkss = std::max(maxkss,k_stepsize);
            }
            if(omega <= k_epsilon)
                break;
        }
        if(verbose)
            std::cout<<"  max stepsize: "<< maxkss<<std::endl;
    }

    void PIS2TASQRTLassoSolver::savetheta() {
        saveVecParam(*theta);
    }

    VectorXd PIS2TASQRTLassoSolver::predict(int lambdaIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        auto newY = (*design_mat)*(*thetas[lambdaIdx]);
        return newY;
    }

    VectorXd PIS2TASQRTLassoSolver::predict(const MatrixXd &newX, int lambdaIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        auto newY = newX*(*thetas[lambdaIdx]);
        return newY;
    }

    double PIS2TASQRTLassoSolver::eval(int lambdaIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        auto newY = predict((*design_mat),lambdaIdx);
        auto diff = newY - (*response_vec);
        return diff.norm()/sqrt(1.*ntrain_sample);
    }

    double PIS2TASQRTLassoSolver::eval(const MatrixXd &newX, const VectorXd &targetY, int lambdaIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        auto newY = predict(newX,lambdaIdx);
        auto diff = newY - targetY;
        return diff.norm()/sqrt(1.*targetY.size());
    }

    int PIS2TASQRTLassoSolver::validate(const MatrixXd &newX, const VectorXd &targetY) {
        int optIdx = -1;
        double minloss = std::numeric_limits<double>::max();
        for (int i = 0; i < nlambda; ++i) {
            double newloss = this->eval(newX, targetY, i);
            if(newloss < minloss)
            {
                minloss = newloss;
                optIdx = i;
            }
        }
        return optIdx;
    }

    double PIS2TASQRTLassoSolver::estError(const VectorXd &trueTheta, int lambdaIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        return (trueTheta - (*thetas[lambdaIdx])).norm();
    }


} // namespace fmlbase