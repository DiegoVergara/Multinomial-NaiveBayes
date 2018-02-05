#ifndef MSB_LOCALBINARYPATTERNS_H
#define MSB_LOCALBINARYPATTERNS_H

#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
//#include "LBP.hpp"
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

class MSB_LocalBinaryPatterns
{
public:
    MSB_LocalBinaryPatterns();
    MSB_LocalBinaryPatterns(int _p_block, int _n_features, int _slider, bool _copy_border, bool _multiscale);
    void init(Mat& _image, vector<Rect> _sampleBox);
    void getFeatureValue(Mat& _image, vector<Rect> _sampleBox, bool _isPositiveBox);
    int multiScaleBlock_LBP(Mat& d_img, int y, int x);
    void multiScaleBlock_Image(Mat& d_img);
    vector<float> multiScaleBlock_Mapping();
    MatrixXd sampleFeatureValue, negativeFeatureValue;

private:
    double Integrate(Mat& d_img, int r0, int c0, int r1, int c1);
    bool initialized, copy_border, multiscale;
    int p_blocks, n_features, slider, h_size;
    vector<int> histogram;
};

#endif