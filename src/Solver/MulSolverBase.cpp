// Multivariate Model Base
// Created by haoming on 2018/1/3.
// Description: 
//
#include <fmlbase/MulSolverBase.h>

namespace fmlbase{

    MulSolverBase::MulSolverBase(const utils::FmlParam &param)  {
        solver_param = &param;
        if (param.getStrArg("inputformat") == "csv"){
            fmlbase::utils::readCsvMat(design_mat,param.getStrArg("rootpath")+"/"+param.getStrArg("traindata"));
            fmlbase::utils::readCsvMat(response_mat,param.getStrArg("rootpath")+"/"+param.getStrArg("trainlebel"));
        } else throw std::runtime_error("Input data format: " +param.getStrArg("inputformat")+ "is not supported\n");
        ntrain_sample = response_mat->rows();
        if (ntrain_sample != design_mat->rows())
            throw std::runtime_error("Size of input data and label foes not match\n");
        nfeature = design_mat->cols();
        nresponse = response_mat->cols();
        if(param.hasArg("verbose"))
            verbose = param.getBoolArg("verbose");
        else
            verbose = false;
    }

} // namespace fmlbase