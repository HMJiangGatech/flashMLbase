// PIS2TA CMR Solver
// Created by haoming on 2018/1/3.
// Description: 
//
#ifndef FMLBASE_PIS2TACMRSOLVER_H
#define FMLBASE_PIS2TACMRSOLVER_H


#include <math.h>
#include <fmlbase/SolverBase.h>
#include <fmlbase/PIS2TASQRTLassoSolver.h>
#include <fmlbase/utils.h>
#include <limits>       // std::numeric_limits
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace fmlbase{

    class PIS2TACMRSolver : public PIS2TASQRTLassoSolver {
        explicit PIS2TACMRSolver(const utils::FmlParam &param);
        void reinitialize() override;


    };
} // namespace fmlbase

#endif //FMLBASE_PIS2TACMRSOLVER_H
