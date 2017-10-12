// PISTA Lasso Solver
// Created by haoming on 2017/10/10.
// Description: 
//
#ifndef FMLBASE_PISTALASSOSOLVER_H
#define FMLBASE_PISTALASSOSOLVER_H

#include <math.h>
#include <fmlbase/PIS2TASQRTLassoSolver.h>
#include <fmlbase/utils.h>
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace fmlbase{
    class PISTALassoSolver: public PIS2TASQRTLassoSolver {
    public:
        explicit PISTALassoSolver(const utils::FmlParam &param);
        void reinitialize() override;

        // return the value of loss function
        inline double loss_value(VectorXd *theta_t) override {
            double objval;
            objval = ((*response_vec) - (*design_mat)*(*theta_t)).squaredNorm() / (2.*ntrain_sample);
            return objval;
        }
        inline double loss_value() override {
            double objval;
            objval = ((*response_vec) - (*design_mat)*(*theta)).squaredNorm() / (2.*ntrain_sample);
            return objval;
        }

        // return the value of objective function
        inline double obj_value(VectorXd *theta_t) override {
            double objval;
            objval = ((*response_vec) - (*design_mat)*(*theta_t)).squaredNorm() / (2.*ntrain_sample) + lambda*(*theta_t).cwiseAbs().sum();
            return objval;
        }
        inline double obj_value() override {
            double objval;
            objval = ((*response_vec) - (*design_mat)*(*theta)).squaredNorm() / (2.*ntrain_sample) + lambda*(*theta).cwiseAbs().sum();
            return objval;
        }

        // return the gradient of total objective function
        inline void loss_grad(VectorXd &grad, VectorXd *theta_t) override {
            grad = (*design_mat).transpose()* ((*design_mat)*(*theta_t) - (*response_vec));
            grad /= (1.*ntrain_sample);
        }
        inline void loss_grad(VectorXd &grad) override {
            grad = (*design_mat).transpose()* ((*design_mat)*(*theta) - (*response_vec));
            grad /= (1.*ntrain_sample);
        }

        // return the gradient of total objective function with sub-gradient taking 0 at 0.
        inline void obj_grad(VectorXd &grad, VectorXd *theta_t) override {
            auto residue = ((*design_mat)*(*theta_t) - (*response_vec));
            grad = (*design_mat).transpose()* residue;
            grad /= (1.*ntrain_sample);
            grad += lambda*theta_t->cwiseSign();
        }
        inline void obj_grad(VectorXd &grad) override {
            auto residue = ((*design_mat)*(*theta) - (*response_vec));
            grad = (*design_mat).transpose()* residue;
            grad /= (1.*ntrain_sample);
            grad += lambda*theta->cwiseSign();
        }

        inline MatrixXd hessian() override {
            if(this->hessianMat != nullptr)
                return *this->hessianMat;

            this->hessianMat = new MatrixXd(design_mat->transpose() * (*design_mat) / (1.*ntrain_sample));
            return *this->hessianMat;
        }

    private:
        MatrixXd *hessianMat;
        double sigma;       // the hyperparameter of the variance of the noise
    };
} // namespace fmlbase



#endif //FMLBASE_PISTALASSOSOLVER_H
