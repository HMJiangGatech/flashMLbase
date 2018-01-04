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


} // namespace fmlbase

