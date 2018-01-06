// PIS2TA SQRT Lasso Solver (Pathwise Optimization Iterative Shrinkage Thresholding)
// Created by haoming on 2017/10/5.
// Description: 
//
#ifndef FMLBASE_PIS2TASQRTLASSOSOLVER_H
#define FMLBASE_PIS2TASQRTLASSOSOLVER_H

#include <math.h>
#include <fmlbase/SolverBase.h>
#include <fmlbase/utils.h>
#include <limits>       // std::numeric_limits
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace fmlbase{

    class PIS2TASQRTLassoSolver : public SolverBase {
    public:
        explicit PIS2TASQRTLassoSolver(const utils::FmlParam &param);
        virtual void initialize();
        ~PIS2TASQRTLassoSolver();
        virtual void reinitialize();

        // return the value of q function
        inline double q_value(const VectorXd &temp_theta, const VectorXd &grad, const double &tilde_stepsize, const double loss){
            double q_val;
            auto diff_theta = (temp_theta.array() - theta->array()).matrix();
            q_val = loss + grad.transpose()*diff_theta + 0.5*tilde_stepsize*diff_theta.squaredNorm()
                    + lambda * temp_theta.cwiseAbs().sum();
            return q_val;
        }

        // return the residue
        virtual inline void get_residure(VectorXd &residure, VectorXd *theta_t){
            residure = (*response_vec) - (*design_mat)*(*theta_t);
        }
        virtual inline void get_residure(VectorXd &residure){
            residure = (*response_vec) - (*design_mat)*(*theta);
        }

        // return the value of loss function
        virtual inline double loss_value(VectorXd *theta_t){
            double objval;
            objval = ((*response_vec) - (*design_mat)*(*theta_t)).norm() / sqrt(1.*ntrain_sample);
            return objval;
        }
        virtual inline double loss_value(){
            double objval;
            objval = ((*response_vec) - (*design_mat)*(*theta)).norm() / sqrt(1.*ntrain_sample);
            return objval;
        }

        // return the value of objective function
        virtual inline double obj_value(VectorXd *theta_t){
            double objval;
            objval = ((*response_vec) - (*design_mat)*(*theta_t)).norm() / sqrt(1.*ntrain_sample) + lambda*(*theta_t).cwiseAbs().sum();
            return objval;
        }
        virtual inline double obj_value(){
            double objval;
            objval = ((*response_vec) - (*design_mat)*(*theta)).norm() / sqrt(1.*ntrain_sample) + lambda*(*theta).cwiseAbs().sum();
            return objval;
        }

        // return the gradient of total objective function
        virtual inline void loss_grad(VectorXd &grad, VectorXd *theta_t){
            auto residue = ((*design_mat)*(*theta_t) - (*response_vec));
            grad = (*design_mat).transpose()* residue;
            grad /= sqrt(1.*ntrain_sample)*residue.norm();
        }
        virtual inline void loss_grad(VectorXd &grad){
            auto residue = ((*design_mat)*(*theta) - (*response_vec));
            grad = (*design_mat).transpose()* residue;
            grad /= sqrt(1.*ntrain_sample)*residue.norm();
        }
        virtual inline double loss_a_grad(VectorXd &grad){
            auto residue = ((*design_mat)*(*theta) - (*response_vec));
            grad = (*design_mat).transpose()* residue;
            grad /= sqrt(1.*ntrain_sample)*residue.norm();
            return residue.norm()/sqrt(1.*ntrain_sample);
        }
        // return the gradient of total objective function with sub-gradient taking 0 at 0.
        virtual inline void obj_grad(VectorXd &grad, VectorXd *theta_t){
            auto residue = ((*design_mat)*(*theta_t) - (*response_vec));
            grad = (*design_mat).transpose()* residue;
            grad /= sqrt(1.*ntrain_sample)*residue.norm();
            grad += lambda*theta_t->cwiseSign();
        }
        virtual inline void obj_grad(VectorXd &grad){
            auto residue = ((*design_mat)*(*theta) - (*response_vec));
            grad = (*design_mat).transpose()* residue;
            grad /= sqrt(1.*ntrain_sample)*residue.norm();
            grad += lambda*theta->cwiseSign();
        }

        virtual inline MatrixXd hessian(){
            auto residue =  ((*design_mat)*(*theta) - (*response_vec));
            double residue_norm = residue.norm();
            auto temp_vec = (*design_mat).transpose()*residue;
            auto hessianMat = 1./(residue_norm*sqrt(1.*ntrain_sample)) * ((design_mat->transpose() * (*design_mat)) - (temp_vec * temp_vec.transpose())/ pow(residue_norm,2));
            return hessianMat;
        }

        virtual inline double hessian_norm(){
            auto residue =  ((*design_mat)*(*theta) - (*response_vec));
            double residue_norm = residue.norm();
            auto temp_vec = (*design_mat).transpose()*residue;
            auto hessianMat = 1./(residue_norm*sqrt(1.*ntrain_sample)) * ((design_mat->transpose() * (*design_mat)) - (temp_vec * temp_vec.transpose())/ pow(residue_norm,2));
            return hessianMat.norm();
        }

        void train() override;
        virtual void ISTA(double k_stepsize, double k_epsilon);


        VectorXd predict(int lambdaIdx = -1); // for training data
        VectorXd predict(const MatrixXd &newX, int lambdaIdx = -1);

        // residue norm / sqrt(n)
        double eval(int lambdaIdx = -1); // for training data
        double eval(const MatrixXd &newX, const VectorXd &targetY, int lambdaIdx = -1);

        // estimation error
        double estError(const VectorXd &trueTheta, int lambdaIdx = -1);

        // return the index of the best lambda
        int validate(const MatrixXd &newX, const VectorXd &targetY);

        inline VectorXd gettheta(){
            return *(thetas.back());
        }
        void savetheta();

    public:
        VectorXd* theta;                // The model parameter.
        double lambda;                  // The regularization parameter
        int niter;                      // The number of iterations
        int &nlambda;                    // The number of lambdas
        std::vector<VectorXd*> thetas;   // The model parameters on the path.
        double* lambdas;                // The regularization parameters sequence on the path.
        double stepsize_max;            // The max step size parameter for the inner loop (backtracking line search)
        double epsilon;                 // The specified precision
    };
} // namespace fmlbase


#endif //FMLBASE_PIS2TASQRTLASSOSOLVER_H
