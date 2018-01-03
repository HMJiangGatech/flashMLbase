// The base class of solver
// Created by haoming on 2017/10/5.
// Description: 
//
#ifndef FMLBASE_SOLVERBASE_H
#define FMLBASE_SOLVERBASE_H

#include <fmlbase/utils.h>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace fmlbase{

    class SolverBase{
    public:
        SolverBase() = default;
        explicit SolverBase(const utils::FmlParam &param, bool isMulVar = false);
        virtual void train() = 0;
        void saveVecParam(const VectorXd &parameter);
        void saveMatParam(const MatrixXd &parameter);
        virtual ~SolverBase() {
            delete design_mat;
            delete response_vec;
        }

    protected:
        const utils::FmlParam *solver_param;
        MatrixXd* design_mat;
        VectorXd* response_vec;
        long long int ntrain_sample;
        long long int nfeature;
        long long int nresponse;
        bool verbose;
    };
} // namespace fmlbase

#endif //FMLBASE_SOLVERBASE_H
