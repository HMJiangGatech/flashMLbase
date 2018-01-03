// The base class of solver
// Created by haoming on 10/5/17.
//

#include <fmlbase/SolverBase.h>

namespace fmlbase{

    SolverBase::SolverBase(const utils::FmlParam &param) : solver_param(&param) {
        bool isMulVar = false;
        if(param.hasArg("mul_var"))
            if(param.getBoolArg("mul_var"))
                isMulVar = true;
        if (param.getStrArg("inputformat") == "csv"){
            fmlbase::utils::readCsvMat(design_mat,param.getStrArg("rootpath")+"/"+param.getStrArg("traindata"));
            if (!isMulVar)
            {
                fmlbase::utils::readCsvVec(response_vec,param.getStrArg("rootpath")+"/"+param.getStrArg("trainlebel"));
                nresponse = 1;
                ntrain_sample = response_vec->size();
            }
            else
            {
                MatrixXd* response_mat;
                fmlbase::utils::readCsvMat(response_mat,param.getStrArg("rootpath")+"/"+param.getStrArg("trainlebel"));
                response_vec = new  VectorXd(Eigen::Map<VectorXd>(response_mat->data(), response_mat->cols()*response_mat->rows()));
                nresponse = response_mat->cols();
                ntrain_sample = response_mat->rows();
                delete response_mat;
            }
        } else throw std::runtime_error("Input data format: " +param.getStrArg("inputformat")+ "is not supported\n");

        if (ntrain_sample != design_mat->rows())
            throw std::runtime_error("Size of input data and label foes not match\n");
        nfeature = design_mat->cols();
        if(param.hasArg("verbose"))
            verbose = param.getBoolArg("verbose");
        else
            verbose = false;
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