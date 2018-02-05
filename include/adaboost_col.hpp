#ifndef ADABOOST_H
#define ADABOOST_H

#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
//#include "multinomialnaivebayes.hpp"
#include "gaussiannaivebayes_col.hpp"
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <vector>
#include <float.h>
#include <math.h>
 
using namespace std;
using namespace Eigen;


class Adaboost
{
public:
    Adaboost();
    Adaboost(int n_estimators, double alpha, double learning_rate);
    void fit(MatrixXd &dX, VectorXi &lX);
    VectorXi predict( MatrixXd &dY);
    //double boost(VectorXd &w, int iteration, VectorXd &errors, vector<GaussianNaiveBayes> &classifiers);
    double boost(VectorXd &w, int iteration, VectorXd &errors, vector<GaussianNaiveBayes> &classifiers, VectorXi &columns);
    MatrixXd *getdata();
    MatrixXd *gettest();
    VectorXi *getlabels(); 
private:
    int n_estimators, n_data, dim, n_data_test, n_classes;
    VectorXd alphas;
    VectorXi columns;
    vector<GaussianNaiveBayes> classifiers;
    //std::map<unsigned int,GaussianNaiveBayes, int> classifiers;
    double alpha, learning_rate;
    mt19937 generator;
    MatrixXd *dX, *dY;
    VectorXi *lX;
    vector <int> classes;
    bool initialized;
};

#endif