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

        // return the value of loss function
        inline double loss_value(VectorXd *theta_t) override {
            double objval = 0;
            double  sqrt_ntrain_sample_inv = 1. / sqrt(1. * ntrain_sample);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                objval += ((response_vec->segment(i * nfeature, nfeature)) -
                           (*design_mat) * (theta_t->segment(i * nfeature, nfeature))).norm() * sqrt_ntrain_sample_inv;
            }
            return objval;
        }

        inline double loss_value() override {
            double objval =0;
            double  sqrt_ntrain_sample_inv = 1. / sqrt(1. * ntrain_sample);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                objval += ((response_vec->segment(i * nfeature, nfeature)) -
                           (*design_mat) * (theta->segment(i * nfeature, nfeature))).norm() * sqrt_ntrain_sample_inv;
            }
            return objval;
        }

        // return the value of objective function
        inline double obj_value(VectorXd *theta_t) override {
            double objval =0;
            VectorXd regvec = VectorXd::Zero(nfeature);
            double  sqrt_ntrain_sample_inv = 1. / sqrt(1. * ntrain_sample);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                objval += ((response_vec->segment(i * nfeature, nfeature)) -
                           (*design_mat) * (theta_t->segment(i * nfeature, nfeature))).norm() *
                          sqrt_ntrain_sample_inv;
                regvec += theta->segment(i * nfeature, nfeature).array().matrix();
            }

            objval += lambda*regvec.cwiseSqrt().sum();
            return objval;
        }
        inline double obj_value() override {
            double objval =0;
            VectorXd regvec = VectorXd::Zero(nfeature);
            double  sqrt_ntrain_sample_inv = 1. / sqrt(1. * ntrain_sample);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                objval += ((response_vec->segment(i * nfeature, nfeature)) -
                           (*design_mat) * (theta->segment(i * nfeature, nfeature))).norm() *
                          sqrt_ntrain_sample_inv;
                regvec += theta->segment(i * nfeature, nfeature).array().matrix();
            }

            objval += lambda*regvec.cwiseSqrt().sum();
            return objval;
        }

        // return the gradient of total objective function
        inline void loss_grad(VectorXd &grad, VectorXd *theta_t) override {
            grad = VectorXd::Zero(nparameter);
            double  sqrt_ntrain_sample = sqrt(1. * ntrain_sample);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                auto residue = ((*design_mat) * (theta_t->segment(i * nfeature, nfeature)) -
                                (response_vec->segment(i * nfeature, nfeature)));
                grad.segment(i * nfeature, nfeature) = (*design_mat).transpose() * residue;
                grad.segment(i * nfeature, nfeature) /= sqrt_ntrain_sample * residue.norm();
            }
        }
        inline void loss_grad(VectorXd &grad) override {
            grad = VectorXd::Zero(nparameter);
            double  sqrt_ntrain_sample = sqrt(1. * ntrain_sample);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                auto residue = ((*design_mat) * (theta->segment(i * nfeature, nfeature)) -
                                (response_vec->segment(i * nfeature, nfeature)));
                grad.segment(i * nfeature, nfeature) = (*design_mat).transpose() * residue;
                grad.segment(i * nfeature, nfeature) /= sqrt_ntrain_sample * residue.norm();
            }
        }
        inline double loss_a_grad(VectorXd &grad) override {
            double objval = 0;
            grad = VectorXd::Zero(nparameter);
            double  sqrt_ntrain_sample = sqrt(1. * ntrain_sample);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                auto residue = ((*design_mat) * (theta->segment(i * nfeature, nfeature)) -
                                (response_vec->segment(i * nfeature, nfeature)));
                grad.segment(i * nfeature, nfeature) = (*design_mat).transpose() * residue;
                grad.segment(i * nfeature, nfeature) /= sqrt_ntrain_sample * residue.norm();
                objval += residue.norm() / sqrt_ntrain_sample;
            }
            return objval;
        }

        // return the gradient of total objective function with sub-gradient taking 0 at 0.
        inline void obj_grad(VectorXd &grad, VectorXd *theta_t) override {
            grad = VectorXd::Zero(nparameter);
            double  sqrt_ntrain_sample = sqrt(1. * ntrain_sample);
            VectorXd regvec = VectorXd::Zero(nfeature);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                auto residue = ((*design_mat) * (theta_t->segment(i * nfeature, nfeature)) -
                                (response_vec->segment(i * nfeature, nfeature)));
                grad.segment(i * nfeature, nfeature) = (*design_mat).transpose() * residue;
                grad.segment(i * nfeature, nfeature) /= sqrt_ntrain_sample * residue.norm();
                regvec += theta->segment(i * nfeature, nfeature).array().matrix();
            }
            regvec = lambda*regvec.cwiseSqrt();
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                grad.segment(i * nfeature, nfeature) += theta_t->cwiseQuotient(regvec);
            }
        }

        inline void obj_grad(VectorXd &grad) override {
            grad = VectorXd::Zero(nparameter);
            double  sqrt_ntrain_sample = sqrt(1. * ntrain_sample);
            VectorXd regvec = VectorXd::Zero(nfeature);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                auto residue = ((*design_mat) * (theta->segment(i * nfeature, nfeature)) -
                                (response_vec->segment(i * nfeature, nfeature)));
                grad.segment(i * nfeature, nfeature) = (*design_mat).transpose() * residue;
                grad.segment(i * nfeature, nfeature) /= sqrt_ntrain_sample * residue.norm();
                regvec += theta->segment(i * nfeature, nfeature).array().matrix();
            }
            regvec = lambda*regvec.cwiseSqrt();
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                grad.segment(i * nfeature, nfeature) += theta->cwiseQuotient(regvec);
            }
        }

        inline double hessian_norm() override {
            

            auto residue =  ((*design_mat)*(*theta) - (*response_vec));
            double residue_norm = residue.norm();
            auto temp_vec = (*design_mat).transpose()*residue;
            auto hessianMat = 1./(residue_norm*sqrt(1.*ntrain_sample)) * ((design_mat->transpose() * (*design_mat)) - (temp_vec * temp_vec.transpose())/ pow(residue_norm,2));

            return 0;
        }
    };
} // namespace fmlbase

#endif //FMLBASE_PIS2TACMRSOLVER_H
