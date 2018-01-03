// Multivariate Model Base
// Created by haoming on 2018/1/3.
// Description: 
//
#ifndef FMLBASE_MULSOLVERBASE_H
#define FMLBASE_MULSOLVERBASE_H

#include <fmlbase/SolverBase.h>

namespace fmlbase{

    class MulSolverBase : public SolverBase{
    public:
        MulSolverBase() = default;
        explicit MulSolverBase(const utils::FmlParam &param);
        virtual void train() = 0;

    protected:
        MatrixXd* response_mat;
        int nresponse;
    };
} // namespace fmlbase


#endif //FMLBASE_MULSOLVERBASE_H
