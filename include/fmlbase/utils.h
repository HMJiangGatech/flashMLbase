//
// Created by haoming on 10/4/17.
//

#ifndef FMLBASE_UTILS_H
#define FMLBASE_UTILS_H

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <fstream>
#include <iostream>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;


namespace fmlbase{
    namespace utils{
        /*!
         * \brief base implementation of config reader
         */
        class ConfigReaderBase {
        public:
            /*!
             * \brief get current name, called after Next returns true
             * \return current parameter name
             */
            inline const char *name() const {
                return s_name.c_str();
            }
            /*!
             * \brief get current value, called after Next returns true
             * \return current parameter value
             */
            inline const char *val() const {
                return s_val.c_str();
            }
            /*!
             * \brief move iterator to next position
             * \return true if there is value in next position
             */
            inline bool Next() {
                while (!this->IsEnd()) {
                    GetNextToken(&s_name);
                    if (s_name == "=") return false;
                    return !(GetNextToken(&s_buf) || s_buf != "=")
                           && !(GetNextToken(&s_val) || s_val == "=");
                }
                return false;
            }
            // called before usage
            inline void Init() {
                ch_buf = this->GetChar();
            }

        protected:
            /*!
             * \brief to be implemented by subclass,
             * get next token, return EOF if end of file
             */
            virtual char GetChar() = 0;
            /*! \brief to be implemented by child, check if end of stream */
            virtual bool IsEnd() = 0;

        private:
            char ch_buf;
            std::string s_name, s_val, s_buf;

            inline void SkipLine() {
                do {
                    ch_buf = this->GetChar();
                } while (ch_buf != EOF && ch_buf != '\n' && ch_buf != '\r');
            }

            inline void ParseStr(std::string *tok) {
                while ((ch_buf = this->GetChar()) != EOF) {
                    switch (ch_buf) {
                        case '\\': *tok += this->GetChar(); break;
                        case '\"': return;
                        case '\r':
                        case '\n': throw std::runtime_error ("ConfigReader: unterminated string\n");
                        default: *tok += ch_buf;
                    }
                }
                throw std::runtime_error ("ConfigReader: unterminated string\n");
            }
            inline void ParseStrML(std::string *tok) {
                while ((ch_buf = this->GetChar()) != EOF) {
                    switch (ch_buf) {
                        case '\\': *tok += this->GetChar(); break;
                        case '\'': return;
                        default: *tok += ch_buf;
                    }
                }
                throw std::runtime_error ("unterminated string\n");
            }
            // return newline
            inline bool GetNextToken(std::string *tok) {
                tok->clear();
                bool new_line = false;
                while (ch_buf != EOF) {
                    switch (ch_buf) {
                        case '#' : SkipLine(); new_line = true; break;
                        case '\"':
                            if (tok->length() == 0) {
                                ParseStr(tok); ch_buf = this->GetChar(); return new_line;
                            } else {
                                throw std::runtime_error ("ConfigReader: token followed directly by string\n");
                            }
                        case '\'':
                            if (tok->length() == 0) {
                                ParseStrML(tok); ch_buf = this->GetChar(); return new_line;
                            } else {
                                throw std::runtime_error ("ConfigReader: token followed directly by string\n");
                            }
                        case '=':
                            if (tok->length() == 0) {
                                ch_buf = this->GetChar();
                                *tok = '=';
                            }
                            return new_line;
                        case '\r':
                        case '\n':
                            if (tok->length() == 0) new_line = true;
                        case '\t':
                        case ' ' :
                            ch_buf = this->GetChar();
                            if (tok->length() != 0) return new_line;
                            break;
                        default:
                            *tok += ch_buf;
                            ch_buf = this->GetChar();
                            break;
                    }
                }
                return tok->length() == 0;
            }
        };
        /*!
         * \brief an iterator use stream base, allows use all types of istream
         */
        class ConfigStreamReader: public ConfigReaderBase {
        public:
            /*!
             * \brief constructor
             * \param fin istream input stream
             */
            explicit ConfigStreamReader(std::istream &fin) : fin(fin) {}

        protected:
            char GetChar() override {
                return static_cast<char>(fin.get());
            }
            /*! \brief to be implemented by child, check if end of stream */
            bool IsEnd() override {
                return fin.eof();
            }

        private:
            std::istream &fin;
        };

        /*!
         * \brief an iterator that iterates over a configure file and gets the configures
         */
        class ConfigIterator: public ConfigStreamReader {
        public:
            /*!
             * \brief constructor
             * \param fname name of configure file
             */
            explicit ConfigIterator(const char *fname) : ConfigStreamReader(fi) {
                fi.open(fname);
                if (fi.fail()) {
                    throw std::runtime_error ("cannot open file \n");
                }
                ConfigReaderBase::Init();
            }
            /*! \brief destructor */
            ~ConfigIterator() {
                fi.close();
            }

        private:
            std::ifstream fi;
        };

        class FmlParam{
        public:
            FmlParam();
            // set parameters from file path
            explicit FmlParam(std::string file_path);

            inline bool hasArg(const std::string &key) const {
                auto  iter = cfg.find(key);
                return !(iter == cfg.end());
            }
            inline std::string getStrArg(const std::string &key) const{
                auto  iter = cfg.find(key);
                if(iter == cfg.end())
                    throw std::runtime_error ("Can not find argument: "+key+"\n");
                return iter->second;
            }
            inline int getIntArg(const std::string &key) const{
                auto  iter = cfg.find(key);
                if(iter == cfg.end())
                    throw std::runtime_error ("Can not find argument: "+key+"\n");
                std::stringstream ss;
                int output;
                ss << iter->second;
                ss >> output;
                return output;
            }
            inline double getDoubleArg(const std::string &key) const{
                auto  iter = cfg.find(key);
                if(iter == cfg.end())
                    throw std::runtime_error ("Can not find argument: "+key+"\n");
                std::stringstream ss;
                double output;
                ss << iter->second;
                ss >> output;
                return output;
            }
            inline bool getBoolArg(const std::string &key) const{
                auto  iter = cfg.find(key);
                if(iter == cfg.end())
                    throw std::runtime_error ("Can not find argument: "+key+"\n");
                std::stringstream ss;
                bool output;
                ss << iter->second;
                ss >> output;
                return output;
            }
        private:
            std::map<std::string, std::string> cfg;
        }; // class FmlParam

        // read eigen matrix from csv file
        void readCsvMat(MatrixXd* &mat, std::string file);
        // read eigen vector from csv file
        void readCsvVec(VectorXd* &vec, std::string file);
        // write eigen matrix from csv file
        void writeCsvMat(const MatrixXd &mat, std::string file);
        // write eigen vector to csv file
        void writeCsvVec(const VectorXd &vec, std::string file);

    } // namespace utils
} // namespace fmlbase

#endif //FMLBASE_UTILS_H
