//
// Created by haoming on 10/4/17.
//

#include <fmlbase/utils.h>
#include <sstream>

namespace fmlbase{

    namespace utils
    {
        FmlParam::FmlParam() {
            std::cout << "Warning: empty initialization!!\n";
        }

        FmlParam::FmlParam(std::string file_path) {
            std::string config_path;
            config_path = file_path + "/config.txt";
            cfg.emplace("rootpath", file_path);
            std::ifstream config_in(config_path.c_str());
            ConfigIterator itr(config_path.c_str());
            while (itr.Next()) {
                cfg.emplace(std::string(itr.name()),
                                       std::string(itr.val()));
                std::cout << std::string(itr.name()) << ' ' <<std::string(itr.val())<<std::endl;
            }
        }

        void readCsvMat(MatrixXd* &mat, std::string file) {
            std::ifstream fin(file);
            if (!fin.good())
                throw std::runtime_error("Can not open " + file);
            std::vector<std::vector<double>> cache;
            double dbuf;
            char cbuf;
            std::string line_buf;
            while(std::getline(fin, line_buf))
            {
                std::stringstream     lineStream(line_buf);
                std::vector<double>  double_line;
                while (lineStream >> dbuf)
                {
                    double_line.emplace_back(dbuf);
                    if(!(lineStream >> cbuf))
                        break;
                }
                cache.emplace_back(double_line);
            }

            // construct eigen matrix
            int n_row;
            int n_col;
            n_row = cache.size();
            n_col = cache[0].size();
            mat = new  MatrixXd(n_row,n_col);
            for (int i = 0; i < n_row; i++)
                mat->row(i) = VectorXd::Map(&cache[i][0],n_col);
            fin.close();
        }

        void readCsvVec(VectorXd *&vec, std::string file) {
            std::ifstream fin(file);
            if (!fin.good())
                throw std::runtime_error("Can not open " + file);
            std::vector<double> cache;
            double dbuf;
            char cbuf;
            std::string line_buf;
            while(std::getline(fin, line_buf))
            {
                std::stringstream     lineStream(line_buf);
                while (lineStream >> dbuf)
                {
                    cache.emplace_back(dbuf);
                    if(!(lineStream >> cbuf))
                        break;
                }
            }
            int n_elm;
            n_elm = cache.size();
            vec = new  VectorXd(n_elm);
            *vec = VectorXd::Map(&cache[0],n_elm);
            fin.close();
        }

        void writeCsvMat(const MatrixXd &mat, std::string file) {
            std::ofstream fout(file);
            if (!fout.good())
                throw std::runtime_error("Can not open " + file);
            std::stringstream ss;
            std::string s;
            ss << mat;
            s = ss.str();
            std::replace( s.begin(), s.end(), ' ', ',');
            fout << s;
            fout.close();
        }

        void writeCsvVec(const VectorXd &vec, std::string file) {
            std::ofstream fout(file);
            if (!fout.good())
                throw std::runtime_error("Can not open " + file);
            fout << vec;
            fout.close();
        }

    } // namespace utils
} // namespace fmlbase