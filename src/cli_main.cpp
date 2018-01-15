// The Command Line Interface
// Created by haoming on 10/4/17.
//


#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <numeric>
#include <chrono>

#include <fmlbase/utils.h>
#include <fmlbase/SolverBase.h>
#include <fmlbase/PIS2TASQRTLassoSolver.h>
#include <fmlbase/PISTALassoSolver.h>
#include <fmlbase/PIS2TACMRSolver.h>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

void runCLITask(const std::string &task_path)
{
    fmlbase::utils::FmlParam param(task_path);

    vector<double> times;


    if(param.getStrArg("algorithm") == "sqrtlasso") {
        fmlbase::PIS2TASQRTLassoSolver solver(param);
        solver.initialize();
        MatrixXd* testx;
        fmlbase::utils::readCsvMat(testx, param.getStrArg("rootpath") + "/"+param.getStrArg("testdata"));
        VectorXd* testy;
        fmlbase::utils::readCsvVec(testy, param.getStrArg("rootpath") + "/"+param.getStrArg("testlabel"));

        for (int i = 0; i < param.getIntArg("nexp"); ++i) {
            solver.reinitialize();
            auto begin = std::chrono::steady_clock::now();
            solver.train();
            auto end = std::chrono::steady_clock::now();
            auto diff = 1. * (end - begin).count() * nanoseconds::period::num / nanoseconds::period::den;
            times.emplace_back(diff);
            cout << i << "th trail, training time (/s): " << diff << endl;
            cout << i << "train error: " << solver.eval() << endl;
            cout << i << "test error: " << solver.eval(*testx, *testy) << endl;
        }
    }
    if(param.getStrArg("algorithm") == "lasso") {
        fmlbase::PISTALassoSolver solver(param);
        solver.initialize();
        MatrixXd* testx;
        fmlbase::utils::readCsvMat(testx, param.getStrArg("rootpath") + "/"+param.getStrArg("testdata"));
        VectorXd* testy;
        fmlbase::utils::readCsvVec(testy, param.getStrArg("rootpath") + "/"+param.getStrArg("testlabel"));

        for (int i = 0; i < param.getIntArg("nexp"); ++i) {
            solver.reinitialize();
            auto begin = std::chrono::steady_clock::now();
            solver.train();
            auto end = std::chrono::steady_clock::now();
            auto diff = 1. * (end - begin).count() * nanoseconds::period::num / nanoseconds::period::den;
            times.emplace_back(diff);
            cout << i << "th trail, training time (/s): " << diff << endl;
            cout << i << "train error: " << solver.eval() << endl;
            cout << i << "test error: " << solver.eval(*testx, *testy) << endl;
        }
    }
    if(param.getStrArg("algorithm") == "CMR") {
        fmlbase::PIS2TACMRSolver solver(param);
        solver.initialize();
        MatrixXd* testx;
        fmlbase::utils::readCsvMat(testx, param.getStrArg("rootpath") + "/"+param.getStrArg("testdata"));
        MatrixXd* testy;
        fmlbase::utils::readCsvMat(testy, param.getStrArg("rootpath") + "/"+param.getStrArg("testlabel"));

        for (int i = 0; i < param.getIntArg("nexp"); ++i) {
            solver.reinitialize();
            auto begin = std::chrono::steady_clock::now();
            solver.train();
            auto end = std::chrono::steady_clock::now();
            auto diff = 1. * (end - begin).count() * nanoseconds::period::num / nanoseconds::period::den;
            times.emplace_back(diff);
            cout << i << "th trail, training time (/s): " << diff << endl;
            cout << i << "train error: " << solver.eval() << endl;
            cout << i << "test error: " << solver.eval(*testx, *testy) << endl;
        }
    }
    if(param.getStrArg("algorithm") == "SPME") {
        MatrixXd* S;
        fmlbase::utils::readCsvMat(S, param.getStrArg("rootpath") + "/"+param.getStrArg("data"));
        auto nfeature = S->rows();
        vector<fmlbase::PIS2TASQRTLassoSolver*> solver_vec;
        for (int j = 0; j < nfeature; ++j) {
            auto temp_row = S->row(0);
            S->row(0) = S->row(j);
            S->row(j) = temp_row;
            fmlbase::PIS2TASQRTLassoSolver *new_solver = new fmlbase::PIS2TASQRTLassoSolver(param, S->bottomRows(nfeature-1), S->row(0));
            new_solver->initialize();
            solver_vec.push_back(new_solver);
        }

        for (int i = 0; i < param.getIntArg("nexp"); ++i) {
            for (int j = 0; j < nfeature; ++j)
                solver_vec[j]->reinitialize();
            auto begin = std::chrono::steady_clock::now();
            for (int j = 0; j < nfeature; ++j)
                solver_vec[j]->train();
            auto end = std::chrono::steady_clock::now();
            auto diff = 1. * (end - begin).count() * nanoseconds::period::num / nanoseconds::period::den;
            times.emplace_back(diff);
            cout << i << "th trail, training time (/s): " << diff << endl;
        }
        int non_zeros = 0;
        for (int j = 0; j < nfeature; ++j)
            non_zeros += ((*(solver_vec[j]->theta)).array()==0).sum();
        cout << "Sparsity: " << 1.*non_zeros/nfeature/nfeature << endl;

    }

    double sumT = 0;
    for(double t : times)
        sumT += t;

    cout<<"mean training time (/s): "<<1.0*sumT/param.getIntArg("nexp")<<endl;


}

int main(int argc, const char * argv[])
{

    if(argc < 2)
        runCLITask("./Tasks/Estrogen");
    else
        runCLITask(argv[1]);

    return 0;
}
