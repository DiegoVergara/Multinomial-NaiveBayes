#ifndef LBP_DATASET_H
#define LBP_DATASET_H

#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
#include "LBP.hpp"
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <vector>
#include <float.h>
#include <math.h>
 
using namespace std;
using namespace Eigen;
using namespace lbp;

class LBP_dataset
{
public:
    LBP_dataset();
    LBP_dataset(string image_path, string _fn_csv, int _rad, int _pts, int _subi, string _mapping, bool _normalizeHist);
    void init();
    void create_in_file(string _output, string _type);
    void get_dataset(MatrixXd &_data, VectorXi &_label, string _type);

private:
    bool initialized;
    int translate(string dictionary[], string data_class, string target);
    string fn_csv, output_path, image_path;
    string *v_gender;
    string *v_age;
    int rad; // Radius (default=1)
    int pts; // Number of support points (default=8)
    string mapping; //Mapping choose between: (default=none) "u2" "ri" "riu2" "hf"
    int subi; //Sub blocks = subi*subi
    bool normalizeHist;
    vector<vector<double>> data;
    vector<int> labels_age;
    vector<int> labels_gender;
    int n_gender, n_age;
};

#endif