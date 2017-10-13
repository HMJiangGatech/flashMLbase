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
    fmlbase::PIS2TASQRTLassoSolver solver1(param);
    fmlbase::PISTALassoSolver solver2(param);

    {
        for (int i = 0; i < param.getIntArg("nexp"); ++i) {
            if(param.getStrArg("algorithm") == "sqrtlasso")
            {
                solver1.reinitialize();
                auto begin = std::chrono::steady_clock::now();
                solver1.train();
                auto end = std::chrono::steady_clock::now();
                auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
                times1.emplace_back(diff);
                //cout<<i<<"th trail, training time (/s): "<<diff<<endl;

            }
            if(param.getStrArg("algorithm") == "lasso")
            {
                solver2.reinitialize();
                auto begin = std::chrono::steady_clock::now();
                solver2.train();
                auto end = std::chrono::steady_clock::now();
                auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
                times2.emplace_back(diff);
                cout<<i<<"th trail, training time (/s): "<<diff<<endl;
            }
        }

        if(param.getStrArg("algorithm") == "sqrtlasso")
        {
            double mean = std::accumulate(std::begin(times1), std::end(times1), 0.0)/times1.size();
            double accum  = 0.0;
            std::for_each (std::begin(times1), std::end(times1), [&](const double d) {
                accum  += (d-mean)*(d-mean);
            });
            double stdev = sqrt(accum/(times1.size()-1)); //standard deviation
            cout.setf(ios::fixed);
            std::cout<<"mean training time (/s): "<<setprecision(4)<<mean
                     <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
        }

        if(param.getStrArg("algorithm") == "lasso")
        {
            double mean = std::accumulate(std::begin(times2), std::end(times2), 0.0)/times2.size();
            double accum  = 0.0;
            std::for_each (std::begin(times2), std::end(times2), [&](const double d) {
                accum  += (d-mean)*(d-mean);
            });
            double stdev = sqrt(accum/(times2.size()-1)); //standard deviation
            cout.setf(ios::fixed);
            std::cout<<"mean training time (/s): "<<setprecision(4)<<mean
                     <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
        }

    }


    solver1.epsilon *= 0.1;
    times1.clear();
    {
        for (int i = 0; i < param.getIntArg("nexp"); ++i) {
            if(param.getStrArg("algorithm") == "sqrtlasso")
            {
                solver1.reinitialize();
                auto begin = std::chrono::steady_clock::now();
                solver1.train();
                auto end = std::chrono::steady_clock::now();
                auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
                times1.emplace_back(diff);
                //cout<<i<<"th trail, training time (/s): "<<diff<<endl;

            }
            if(param.getStrArg("algorithm") == "lasso")
            {
                solver2.reinitialize();
                auto begin = std::chrono::steady_clock::now();
                solver2.train();
                auto end = std::chrono::steady_clock::now();
                auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
                times2.emplace_back(diff);
                cout<<i<<"th trail, training time (/s): "<<diff<<endl;
            }
        }

        if(param.getStrArg("algorithm") == "sqrtlasso")
        {
            double mean = std::accumulate(std::begin(times1), std::end(times1), 0.0)/times1.size();
            double accum  = 0.0;
            std::for_each (std::begin(times1), std::end(times1), [&](const double d) {
                accum  += (d-mean)*(d-mean);
            });
            double stdev = sqrt(accum/(times1.size()-1)); //standard deviation
            cout.setf(ios::fixed);
            std::cout<<"mean training time (/s): "<<setprecision(4)<<mean
                     <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
        }

        if(param.getStrArg("algorithm") == "lasso")
        {
            double mean = std::accumulate(std::begin(times2), std::end(times2), 0.0)/times2.size();
            double accum  = 0.0;
            std::for_each (std::begin(times2), std::end(times2), [&](const double d) {
                accum  += (d-mean)*(d-mean);
            });
            double stdev = sqrt(accum/(times2.size()-1)); //standard deviation
            cout.setf(ios::fixed);
            std::cout<<"mean training time (/s): "<<setprecision(4)<<mean
                     <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
        }

    }

    solver1.epsilon *= 0.1;
    times1.clear();
    {
        for (int i = 0; i < param.getIntArg("nexp"); ++i) {
            if(param.getStrArg("algorithm") == "sqrtlasso")
            {
                solver1.reinitialize();
                auto begin = std::chrono::steady_clock::now();
                solver1.train();
                auto end = std::chrono::steady_clock::now();
                auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
                times1.emplace_back(diff);
                //cout<<i<<"th trail, training time (/s): "<<diff<<endl;

            }
            if(param.getStrArg("algorithm") == "lasso")
            {
                solver2.reinitialize();
                auto begin = std::chrono::steady_clock::now();
                solver2.train();
                auto end = std::chrono::steady_clock::now();
                auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
                times2.emplace_back(diff);
                cout<<i<<"th trail, training time (/s): "<<diff<<endl;
            }
        }

        if(param.getStrArg("algorithm") == "sqrtlasso")
        {
            double mean = std::accumulate(std::begin(times1), std::end(times1), 0.0)/times1.size();
            double accum  = 0.0;
            std::for_each (std::begin(times1), std::end(times1), [&](const double d) {
                accum  += (d-mean)*(d-mean);
            });
            double stdev = sqrt(accum/(times1.size()-1)); //standard deviation
            cout.setf(ios::fixed);
            std::cout<<"mean training time (/s): "<<setprecision(4)<<mean
                     <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
        }

        if(param.getStrArg("algorithm") == "lasso")
        {
            double mean = std::accumulate(std::begin(times2), std::end(times2), 0.0)/times2.size();
            double accum  = 0.0;
            std::for_each (std::begin(times2), std::end(times2), [&](const double d) {
                accum  += (d-mean)*(d-mean);
            });
            double stdev = sqrt(accum/(times2.size()-1)); //standard deviation
            cout.setf(ios::fixed);
            std::cout<<"mean training time (/s): "<<setprecision(4)<<mean
                     <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
        }

    }

}

int main(int argc, const char * argv[])
{

    if(argc < 2)
        runCLITask("./Tasks/Synthetic");
    else
        runCLITask(argv[1]);

    return 0;
}
