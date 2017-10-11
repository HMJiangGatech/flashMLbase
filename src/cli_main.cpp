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

    if(param.getStrArg("algorithm") == "sqrtlasso")
    {
        fmlbase::PIS2TASQRTLassoSolver solver(param);
        vector<double> times;
        vector<double> est_error;
        VectorXd *trueTheta;
        fmlbase::utils::readCsvVec(trueTheta, param.getStrArg("rootpath")+"/"+param.getStrArg("truetheta"));

        for (int i = 0; i < 1; ++i) {
            solver.reinitialize();
            auto begin = std::chrono::steady_clock::now();
            solver.train();
            auto end = std::chrono::steady_clock::now();
            auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
            std::cout<<i<<"th trail, training time (/s): "<<diff<<std::endl;
            times.emplace_back(diff);
        }

        double meantime = std::accumulate(std::begin(times), std::end(times), 0.0)/times.size();
        double accum  = 0.0;
        std::for_each (std::begin(times), std::end(times), [&](const double d) {
            accum  += (d-meantime)*(d-meantime);
        });
        double stdev = sqrt(accum/(times.size()-1)); //standard deviation
        cout.setf(ios::fixed);
        std::cout<<"mean training time (/s): "<<setprecision(4)<<meantime
                 <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
        solver.savetheta();
    }
    else if(param.getStrArg("algorithm") == "lasso")
    {
        fmlbase::PISTALassoSolver solver(param);
        vector<double> times;
        vector<double> est_error;
        VectorXd *trueTheta;
        fmlbase::utils::readCsvVec(trueTheta, param.getStrArg("rootpath")+"/"+param.getStrArg("truetheta"));

        for (int i = 0; i < 1; ++i) {
            solver.reinitialize();
            auto begin = std::chrono::steady_clock::now();
            solver.train();
            auto end = std::chrono::steady_clock::now();
            auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
            std::cout<<i<<"th trail, training time (/s): "<<diff<<std::endl;
            times.emplace_back(diff);
        }

        double meantime = std::accumulate(std::begin(times), std::end(times), 0.0)/times.size();
        double accum  = 0.0;
        std::for_each (std::begin(times), std::end(times), [&](const double d) {
            accum  += (d-meantime)*(d-meantime);
        });
        double stdev = sqrt(accum/(times.size()-1)); //standard deviation
        cout.setf(ios::fixed);
        std::cout<<"mean training time (/s): "<<setprecision(4)<<meantime
                 <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
        solver.savetheta();
    }


//    MatrixXd *valX;
//    VectorXd *valY;
//    fmlbase::utils::readCsvMat(valX,param.getStrArg("rootpath")+"/"+param.getStrArg("validationdata"));
//    fmlbase::utils::readCsvVec(valY,param.getStrArg("rootpath")+"/"+param.getStrArg("validationlabel"));
//    int optIdx = solver.validate(*valX,*valY);
//
}

int main(int argc, const char * argv[])
{

    if(argc < 2)
        runCLITask("./Tasks/Synthetic");
    else
        runCLITask(argv[1]);

    return 0;
}
