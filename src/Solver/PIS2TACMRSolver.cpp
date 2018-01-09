//
// Created by haoming on 2018/1/3.
// Description: 
//
#include <fmlbase/PIS2TACMRSolver.h>

namespace fmlbase {

    PIS2TACMRSolver::PIS2TACMRSolver(const utils::FmlParam &param) : PIS2TASQRTLassoSolver(param) {
    }
    void PIS2TACMRSolver::reinitialize() {
        PIS2TASQRTLassoSolver::reinitialize();
    }

    VectorXd PIS2TACMRSolver::predict(int lambdaIdx, int responseIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        auto newY = (*design_mat)*((*thetas[lambdaIdx]).segment(responseIdx * nfeature, nfeature));
        return newY;
    }

    VectorXd PIS2TACMRSolver::predict(const MatrixXd &newX, int lambdaIdx, int responseIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        auto newY = newX*((*thetas[lambdaIdx]).segment(responseIdx * nfeature, nfeature));
        return newY;
    }

    double PIS2TACMRSolver::eval(int lambdaIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        double error = 0;
        for (int i = 0; i < nresponse; ++i) {
            auto newY = (*design_mat)*((*thetas[lambdaIdx]).segment(i * nfeature, nfeature));
            auto diff = newY - (response_vec->segment(i * ntrain_sample, ntrain_sample));
            error += diff.squaredNorm();
        }
        return sqrt(error/ntrain_sample);
    }

    double PIS2TACMRSolver::eval(const MatrixXd &newX, const MatrixXd &targetY, int lambdaIdx) {
        if(lambdaIdx == -1)
            lambdaIdx = nlambda-1;
        double error = 0;
        for (int i = 0; i < nresponse; ++i) {
            auto newY = newX*((*thetas[lambdaIdx]).segment(i * nfeature, nfeature));
            auto diff = newY - targetY.col(i);
            error += diff.squaredNorm();
        }
        return sqrt(error/newX.rows());
    }


} // namespace fmlbase

