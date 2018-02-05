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
#include <bitset>
#include <algorithm>
 
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

class LBP_dataset
{
public:
    LBP_dataset();
    LBP_dataset(string image_path, string _fn_csv, int _p_block, int _pts, int _subi, string _mapping, bool _normalizeHist);
    void MB_LBP_image(Mat d_img, int p_blocks, int N_features, int slider, bool borderCopy);
    int MultiScale_LBP(Mat d_img, int x, int y, int p_blocks);
    void init();
    void create_in_file(string _output, string _type);
    void get_dataset(MatrixXd &_data, VectorXi &_label, string _type);
    vector<float> MultiScaleBlock_Mapping();

private:
    bool initialized, mb_initialized;
    int translate(string dictionary[], string data_class, string target);
    double Integrate( Mat d_img, int r0, int c0, int r1, int c1);
    string fn_csv, output_path, image_path;
    string *v_gender;
    string *v_age;
    int p_block; // Radius (default=1)
    int pts; // Number of support points (default=8)
    int neiboneighborhood;
    string mapping; //Mapping choose between: (default=none) "u2" "ri" "riu2" "hf"
    int subi; //Sub blocks = subi*subi
    bool normalizeHist, borderCopy;
    vector<vector<double>> data;
    vector<int> labels_age;
    vector<int> labels_gender;
    vector<int> histogram;
    int n_gender, n_age;
    int h_size, N_features;
};

#endif