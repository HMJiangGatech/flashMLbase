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

            double hessian_norm = 1;
            double  sqrt_ntrain_sample_inv = 1. / sqrt(1. * ntrain_sample);
            #pragma omp parallel for
            for (int i = 0; i < nresponse; ++i) {
                auto subtheta = (theta->segment(i * nfeature, nfeature));
                auto residue = ((response_vec->segment(i * nfeature, nfeature)) -
                 (*design_mat) * subtheta);
                double residue_norm = residue.norm();
                auto temp_vec = (*design_mat).transpose()*residue;
                auto hessianMat = 1./(residue_norm*sqrt(1.*ntrain_sample)) * ((design_mat->transpose() * (*design_mat)) - (temp_vec * temp_vec.transpose())/ pow(residue_norm,2));

                hessian_norm *= hessianMat.norm();
            }

            return hessian_norm;
        }

        // predict
        virtual VectorXd predict(int lambdaIdx, int responseIdx); // for training data
        VectorXd predict(int responseIdx) override { predict(-1,responseIdx); } // for training data
        VectorXd predict() override { throw std::runtime_error("Can not call predict without parameter in CMR\n"); } // for training data
        virtual VectorXd predict(const MatrixXd &newX, int lambdaIdx, int responseIdx);
        VectorXd predict(const MatrixXd &newX, int responseIdx) override { predict(newX, -1, responseIdx); };
        VectorXd predict(const MatrixXd &newX) override { throw std::runtime_error("Can not call predict without without specified responseIdx in CMR\n"); }

        // residue norm / sqrt(n)
        double eval(int lambdaIdx) override; // for training data
        double eval(const MatrixXd &newX, const MatrixXd &targetY, int lambdaIdx);
    };
} // namespace fmlbase

#endif //FMLBASE_PIS2TACMRSOLVER_H
