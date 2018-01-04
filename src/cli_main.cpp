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

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

void runCLITask(const std::string &task_path)
{
    fmlbase::utils::FmlParam param(task_path);

    vector<double> times1;
    vector<double> est_error1;
    vector<double> pred_error1;
    vector<double> times2;
    vector<double> est_error2;
    vector<double> pred_error2;


    if(param.getStrArg("algorithm") == "sqrtlasso")
    {
        fmlbase::PIS2TASQRTLassoSolver solver1(param);
        MatrixXd* testx;
        fmlbase::utils::readCsvMat(testx, param.getStrArg("rootpath") + "/"+param.getStrArg("testdata"));
        VectorXd* testy;
        fmlbase::utils::readCsvVec(testy, param.getStrArg("rootpath") + "/"+param.getStrArg("testlabel"));

        for (int i = 0; i < param.getIntArg("nexp"); ++i)
        {
            solver1.reinitialize();
            auto begin = std::chrono::steady_clock::now();
            solver1.train();
            auto end = std::chrono::steady_clock::now();
            auto diff = 1. * (end - begin).count() * nanoseconds::period::num / nanoseconds::period::den;
            times1.emplace_back(diff);
            cout << i << "th trail, training time (/s): " << diff << endl;
            cout << i << "train error: " << solver1.eval() << endl;
            cout << i << "test error: " << solver1.eval(*testx, *testy) << endl;
        }
    }
    if(param.getStrArg("algorithm") == "lasso")
    {
        fmlbase::PISTALassoSolver solver2(param);
        MatrixXd* testx;
        fmlbase::utils::readCsvMat(testx, param.getStrArg("rootpath") + "/"+param.getStrArg("testdata"));
        VectorXd* testy;
        fmlbase::utils::readCsvVec(testy, param.getStrArg("rootpath") + "/"+param.getStrArg("testlabel"));

        for (int i = 0; i < param.getIntArg("nexp"); ++i)
        {
            solver2.reinitialize();
            auto begin = std::chrono::steady_clock::now();
            solver2.train();
            auto end = std::chrono::steady_clock::now();
            auto diff = 1. * (end - begin).count() * nanoseconds::period::num / nanoseconds::period::den;
            times2.emplace_back(diff);
            cout << i << "th trail, training time (/s): " << diff << endl;
            cout << i << "train error: " << solver2.eval() << endl;
            cout << i << "test error: " << solver2.eval(*testx, *testy) << endl;
        }
    }

    }

    vector<double> &times = times1;
    if(param.getStrArg("algorithm") == "sqrtlasso")
        times = times1;
    if(param.getStrArg("algorithm") == "lasso")
        times = times2;

    double sumT = 0;
    for(double t : times)
        sumT += t;

    cout<<"mean training time (/s): "<<1.0*sumT/param.getIntArg("nexp")<<endl;


}

int main(int argc, const char * argv[])
{

    if(argc < 2)
        runCLITask("./Tasks/DrivFace");
    else
        runCLITask(argv[1]);

    return 0;
}
