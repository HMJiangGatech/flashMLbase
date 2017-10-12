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
//    VectorXd *trueTheta;
//    fmlbase::utils::readCsvVec(trueTheta, param.getStrArg("rootpath")+"/"+param.getStrArg("truetheta"));
//
//    VectorXd *testLabel;
//    fmlbase::utils::readCsvVec(testLabel, param.getStrArg("rootpath")+"/"+param.getStrArg("testlabel"));
//    MatrixXd *testData;
//    fmlbase::utils::readCsvMat(testData, param.getStrArg("rootpath")+"/"+param.getStrArg("testdata"));
//
//    VectorXd *validLabel;
//    fmlbase::utils::readCsvVec(validLabel, param.getStrArg("rootpath")+"/"+param.getStrArg("validationlabel"));
//    MatrixXd *validData;
//    fmlbase::utils::readCsvMat(validData, param.getStrArg("rootpath")+"/"+param.getStrArg("validationdata"));

    vector<double> times1;
    vector<double> est_error1;
    vector<double> pred_error1;
    vector<double> times2;
    vector<double> est_error2;
    vector<double> pred_error2;
    fmlbase::PIS2TASQRTLassoSolver solver1(param);
    fmlbase::PISTALassoSolver solver2(param);

    {
        for (int i = 0; i < 500; ++i) {
            //string runscript = "matlab -wait -nosplash -nodesktop -r \"run Tasks/Synthetic/genData.m; quit();\"";
            //system(runscript.c_str());
            {
                solver1.reinitialize();
                auto begin = std::chrono::steady_clock::now();
                solver1.train();
                auto end = std::chrono::steady_clock::now();
                auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
                times1.emplace_back(diff);
                cout<<i<<"th trail, training time (/s): "<<diff<<endl;
//            int optidx;
//            double cest_error,cpred_error;
//            optidx = solver.validate(*validData,*validLabel);
//            cest_error = solver.estError(*trueTheta,optidx);
//            est_error1.emplace_back(cest_error);
//            cpred_error = solver.eval(*testData,*testLabel);
//            pred_error1.emplace_back(cpred_error);
//
//             <<" | optidx:"<<optidx<<" est_error: "<<cest_error<<" pred_error: "<<cpred_error<<endl;

            }
            {
                solver2.reinitialize();
                auto begin = std::chrono::steady_clock::now();
                solver2.train();
                auto end = std::chrono::steady_clock::now();
                auto diff = 1.*(end - begin).count()*nanoseconds::period::num / nanoseconds::period::den;
                times2.emplace_back(diff);
                cout<<i<<"th trail, training time (/s): "<<diff<<endl;
//
//                int optidx;
//                double cest_error,cpred_error;
//                optidx = solver.validate(*validData,*validLabel);
//                cest_error = solver.estError(*trueTheta,optidx);
//                est_error2.emplace_back(cest_error);
//                cpred_error = solver.eval(*testData,*testLabel);
//                pred_error2.emplace_back(cpred_error);
//
//                cout<<i<<"th trail, training time (/s): "<<diff <<" | optidx:"<<optidx<<" est_error: "<<cest_error<<" pred_error: "<<cpred_error<<endl;

            }
        }

        double mean = std::accumulate(std::begin(times1), std::end(times1), 0.0)/times1.size();
        double accum  = 0.0;
        std::for_each (std::begin(times1), std::end(times1), [&](const double d) {
            accum  += (d-mean)*(d-mean);
        });
        double stdev = sqrt(accum/(times1.size()-1)); //standard deviation
        cout.setf(ios::fixed);
        std::cout<<"mean training time (/s): "<<setprecision(4)<<mean
                 <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;

//
//        mean = std::accumulate(std::begin(est_error1), std::end(est_error1), 0.0)/est_error1.size();
//        accum  = 0.0;
//        std::for_each (std::begin(est_error1), std::end(est_error1), [&](const double d) {
//            accum  += (d-mean)*(d-mean);
//        });
//        stdev = sqrt(accum/(est_error1.size()-1)); //standard deviation
//        cout.setf(ios::fixed);
//        std::cout<<"mean est_error: "<<setprecision(4)<<mean
//                 <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
//
//
//        mean = std::accumulate(std::begin(pred_error1), std::end(pred_error1), 0.0)/pred_error1.size();
//        accum  = 0.0;
//        std::for_each (std::begin(pred_error1), std::end(pred_error1), [&](const double d) {
//            accum  += (d-mean)*(d-mean);
//        });
//        stdev = sqrt(accum/(pred_error1.size()-1)); //standard deviation
//        cout.setf(ios::fixed);
//        std::cout<<"mean pred_error: "<<setprecision(4)<<mean
//                 <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;

        mean = std::accumulate(std::begin(times2), std::end(times2), 0.0)/times2.size();
        accum  = 0.0;
        std::for_each (std::begin(times2), std::end(times2), [&](const double d) {
            accum  += (d-mean)*(d-mean);
        });
        stdev = sqrt(accum/(times2.size()-1)); //standard deviation
        cout.setf(ios::fixed);
        std::cout<<"mean training time (/s): "<<setprecision(4)<<mean
                 <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;

//
//        mean = std::accumulate(std::begin(est_error2), std::end(est_error2), 0.0)/est_error2.size();
//        accum  = 0.0;
//        std::for_each (std::begin(est_error2), std::end(est_error2), [&](const double d) {
//            accum  += (d-mean)*(d-mean);
//        });
//        stdev = sqrt(accum/(est_error2.size()-1)); //standard deviation
//        cout.setf(ios::fixed);
//        std::cout<<"mean est_error: "<<setprecision(4)<<mean
//                 <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
//
//
//        mean = std::accumulate(std::begin(pred_error2), std::end(pred_error2), 0.0)/pred_error2.size();
//        accum  = 0.0;
//        std::for_each (std::begin(pred_error2), std::end(pred_error2), [&](const double d) {
//            accum  += (d-mean)*(d-mean);
//        });
//        stdev = sqrt(accum/(pred_error2.size()-1)); //standard deviation
//        cout.setf(ios::fixed);
//        std::cout<<"mean pred_error: "<<setprecision(4)<<mean
//                 <<" ("<<setprecision(4)<< stdev << ")" <<std::endl;
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
