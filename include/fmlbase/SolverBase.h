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
        explicit SolverBase(const utils::FmlParam &param);
        virtual void train(bool verbose) = 0;
        void saveVecParam(const VectorXd &parameter);
        void saveMatParam(const MatrixXd &parameter);

    protected:
        const utils::FmlParam *solver_param;
        MatrixXd* design_mat;
        VectorXd* response_vec;
        int ntrain_sample;
        int nfeature;
    };
} // namespace fmlbase

#endif //FMLBASE_SOLVERBASE_H
