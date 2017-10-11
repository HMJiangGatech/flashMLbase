// The base class of solver
// Created by haoming on 10/5/17.
//

#include <fmlbase/SolverBase.h>

namespace fmlbase{

    SolverBase::SolverBase(const utils::FmlParam &param) : solver_param(&param) {
        if (param.getStrArg("inputformat") == "csv"){
            fmlbase::utils::readCsvMat(design_mat,param.getStrArg("rootpath")+"/"+param.getStrArg("traindata"));
            fmlbase::utils::readCsvVec(response_vec,param.getStrArg("rootpath")+"/"+param.getStrArg("trainlebel"));
        } else throw std::runtime_error("Input data format: " +param.getStrArg("inputformat")+ "is not supported\n");
        ntrain_sample = response_vec->size();
        if (ntrain_sample != design_mat->rows())
            throw std::runtime_error("Size of input data and label foes not match\n");
        nfeature = design_mat->cols();
    }

    void SolverBase::saveMatParam(const MatrixXd &parameter) {
        if (solver_param->getStrArg("outputformat") == "csv"){
            fmlbase::utils::writeCsvMat(parameter,solver_param->getStrArg("rootpath")+"/"+solver_param->getStrArg("savepath_theta"));
        } else throw std::runtime_error("Output data format: " +solver_param->getStrArg("outputformat")+ "is not supported\n");
    }

    void SolverBase::saveVecParam(const VectorXd &parameter) {
        if (solver_param->getStrArg("outputformat") == "csv"){
            fmlbase::utils::writeCsvVec(parameter,solver_param->getStrArg("rootpath")+"/"+solver_param->getStrArg("savepath_theta"));
        } else throw std::runtime_error("Output data format: " +solver_param->getStrArg("outputformat")+ "is not supported\n");
    }
} // namespace fmlbase