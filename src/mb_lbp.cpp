
#include "../include/mb_lbp.hpp"

LBP_dataset::LBP_dataset()
{
    initialized=false;
}

LBP_dataset::LBP_dataset(string _image_path, string _fn_csv, int _p_block, int _pts, int _subi, string _mapping, bool _normalizeHist)
{   
    image_path = _image_path;
    fn_csv = _fn_csv;
    //output_path = _output_path;
    p_block = _p_block;
    pts = _pts;
    subi = _subi;
    mapping = _mapping;
    normalizeHist = _normalizeHist;
    n_gender = 3;
    n_age = 28;
    v_gender = new string[n_gender];
    v_gender[0] = "f"; v_gender[1] = "m"; v_gender[2] = "u";
    v_age = new string[n_age];
    v_age[0] = "(0, 2)"; v_age[1] = "(4, 6)"; v_age[2] = "(8, 12)"; v_age[3] = "(8, 23)"; v_age[4] = "(15, 20)";
    v_age[5] = "(25, 32)"; v_age[6] = "(27, 32)"; v_age[7] = "(38, 42)"; v_age[8] = "(38, 43)"; v_age[9] = "(38, 48)";
    v_age[10] = "(48, 53)"; v_age[11] = "(60, 100)"; v_age[12] = "13"; v_age[13] = "2"; v_age[14] = "22"; v_age[15] = "23";
    v_age[16] = "29"; v_age[17] = "3"; v_age[18] = "34"; v_age[19] = "35"; v_age[20] = "36"; v_age[21] = "42"; v_age[22] = "45";
    v_age[23] = "46"; v_age[24] = "55"; v_age[25] = "57"; v_age[26] = "58"; v_age[27] = "None";
    initialized=true;
    try {
        init();
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        initialized=false;
        exit(1);
    }

}

void LBP_dataset::init() {
    if (!initialized) exit(1);
    ifstream csv(fn_csv.c_str());
    char separator = ';';
    if (!csv) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
    string line, path, age, gender;
    //int train_test=0;
    //while (getline(csv, line) && i<max) {
    while (getline(csv, line)) {
        stringstream liness(line);
        path.clear(); age.clear(); gender.clear();
        getline(liness, path, separator);
        getline(liness, age, separator); // age
        getline(liness, gender, separator); // gender
        if(!path.empty() && !age.empty() && !gender.empty()) { 

            
            //int age_label = translate(v_age, "age", age.c_str());
            //int gender_label = translate(v_gender, "gender", gender.c_str());
            
            // 'path' can be file, dir or wildcard path

            String root((image_path+path).c_str());
            vector<String> files;
            glob(root, files, true);
            for(vector<String>::const_iterator f = files.begin(); f != files.end(); ++f) {
                Mat img = imread(*f, IMREAD_GRAYSCALE); // lead image to mat object in grayscale

                //Size size(240, 240);
                //resize(img,img,size); //scale image to size

                int w = img.cols, h = img.rows;
                static bool showSmallSizeWarning = true;
                if(w>0 && h>0 && (w!=img.cols || h!=img.rows)) cout << "\t* Warning: images should be of the same size!" << endl;
                if(showSmallSizeWarning && (img.cols<50 || img.rows<50)) {
                    cout << "* Warning: for better results images should be not smaller than 50x50!" << endl;
                    showSmallSizeWarning = false;
                }

                equalizeHist(img, img); //Equalize Image
                //img.convertTo( img, CV_64F );
                Mat iimg;
                integral(img,iimg);
				        //imwrite( "../data/adience/test/lbp.png", iimg);

				        //cout << iimg << endl;


                //LBP lbp( pts, LBP::strToType( mapping ) );

                Mat R  = (Mat_<int>(30,90) << 224, 249, 100,   8, 224, 249, 100,   8,  46,  53,  46,  57,  57,  50,  49,  52,  48,  57,  57,  55,  51,  95,  57,  56,  52,  50,  48,  48,  48,  48,  51, 102,  95, 111,  46, 106, 112, 103,   0,   8, 208,  78,  32,   0, 232, 154, 218,  71,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  55,  50,  55,  51,  49,  57,  50,  50,  51,  95, 101,  48,  51,
                      97,  53, 102,  56,  52,  97,  98,  95, 111,  46, 106, 112, 103,   0,  85,  79,  32,   0,  89, 179, 231,  72,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  52,  56,  53,  51,  52,  56,  52,  52,  54,  95,  56,  98,  97,  99,  97,  53, 101,  53,  56,  56,  95, 111,  46, 106, 112, 103,   0, 166,  77,  32,   0,  73, 148, 253,  72,  64,   0,   8, 108,
                      97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  57,  46,  49,  48,  51,  57,  57,  52,  48,  52,  48,  55,  52,  95, 100,  97,  53,  53,  52, 102,  57, 101,  98,  54,  95, 111,  46, 106, 112, 103,   0, 123,  80,  32,   0, 199, 106,  33,  73,  68,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  49,  49,  46,  49,  48,
                      57,  52,  51,  53,  53,  54,  52,  49,  52,  95,  56,  49,  98,  52,  55, 102,  56,  50,  55,  53,  95, 111,  46, 106, 112, 103,   0,   0,   0,   8, 159,  79,  32,   0, 219, 124, 155,  73,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  48,  57,  57,  55,  49,  57,  52,  53,  51,  52,  95,  52, 100,  53,  51, 102,  51,  57, 102,  97, 101,  95, 111,  46,
                     106, 112, 103,   0, 227,  77,  32,   0, 236,  75, 168,  73,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  17,  19,   0,   0,  80, 228, 208, 182,  80, 228, 208, 182,   0,   0,   0,   0,   0,   0,   0,   0,  99,  56, 102,  55,  52,  95, 111,  46, 106, 112, 103,   0, 161,  78,  32,   0, 122, 150, 236,  73,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108,
                     105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  51,  46,  49,  49,  55,  50,  55,  51,  50,  51,  50,  49,  51,  95,  52,  50,  56,  48,  50,  48,  52,  52,  99,  57,  95, 111,  46, 106, 112, 103,   0, 118,  79,  32,   0, 154,  83, 151,  74,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  54,  52,  52,  54,  51,  57,  52,  51,  53,  95,  57,
                     100, 100,  54, 102,  50,  50,  55,  50, 102,  95, 111,  46, 106, 112, 103,   0, 249,  78,  32,   0, 136, 254, 158,  74,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  56,  55,  50,  49,  56,  48,  56,  56,  48,  56,  95,  48,  48,  55,  57,  51,  52,  53,  55,  57,  54,  95, 111,  46, 106, 112, 103,   0,   8,  16,  80,  32,   0, 199,  81, 245,  74,  64,   0,
                       8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  48,  52,  53,  51,  55,  57,  54,  53,  50,  54,  95,  51,  98,  49,  50,  49,  51,  98,  99,  49,  98,  95, 111,  46, 106, 112, 103,   0, 218,  79,  32,   0, 125,  36,   7,  75,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,
                      48,  49,  54,  49,  54,  49,  52,  54,  50,  51,  95, 102,  57, 100,  54,  54, 101,  97, 101,  55, 101,  95, 111,  46, 106, 112, 103,   0,  14,  80,  32,   0, 177, 163,  13,  75,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  48,  52,  53,  51,  55,  57,  48,  55,  50,  52,  95,  54,  50, 100, 101,  51,  53,  48,  52,  98, 101,  95, 111,  46, 106, 112,
                     103,   0,   5,  80,  32,   0,  90, 212,  39,  75,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  48,  52,  50,  52,  56,  49,  53,  56,  49,  51,  95, 101,  57,  52,  54,  50,  57,  98,  49, 101,  99,  95, 111,  46, 106, 112, 103,   0,  26,  78,  32,   0, 213, 195,  92,  75,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103,
                     110, 101, 100,  95, 102,  97,  99, 101,  46,  52,  46,  57,  52,  50,  57,  53,  50,  55,  57,  57,  51,  95,  54, 100,  53,  53,  52,  98,  55,  99,  56,  56,  95, 111,  46, 106, 112, 103,   0,   8,  31,  78,  32,   0,  18, 182, 157,  75,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  52,  46,  57,  54,  55,  49,  55,  48,  53,  57,  56,  53,  95,  51,  50,  54,  48,
                      99,  48,  55, 102,  55,  57,  95, 111,  46, 106, 112, 103,   0,   8,  11,  78,  32,   0, 126,   0, 213,  75,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  56,  46,  57,  51,  53,  48,  53,  51,  57,  55,  54,  56,  95, 100,  50,  53,  53,  99,  53,  50, 102,  54,  53,  95, 111,  46, 106, 112, 103,   0,   8, 120,  79,  32,   0,  30, 170, 232,  75,  64,   0,   8, 108,
                      97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  54,  52,  52,  57,  57,  55,  54,  50,  51,  95,  53,  53,  98,  54, 102,  55,  53,  98,  49,  48,  95, 111,  46, 106, 112, 103,   0, 133,  79,  32,   0,  45, 160,  53,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  48,  56,
                      50,  51,  51,  51,  55,  51,  57,  51,  95,  57,  48,  98,  50, 100,  54,  50,  48,  49,  48,  95, 111,  46, 106, 112, 103,   0,  65,  80,  32,   0,  52,  85,  81,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  49,  46,  49,  48,  51,  57,  57,  54,  53,  56,  54,  50,  54,  95,  57,  99,  51,  49,  98,  57,  54,  99,  53, 101,  95, 111,  46, 106, 112, 103,   0,
                     103,  80,  32,   0, 158, 242, 116,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  49,  46,  49,  49,  56,  54,  49,  52,  54,  54,  54,  54,  53,  95,  56,  50, 101,  97,  48,  57,  54,  50,  52,  49,  95, 111,  46, 106, 112, 103,   0,  34,  78,  32,   0, 253, 169, 130,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101,
                     100,  95, 102,  97,  99, 101,  46,  53,  46,  49,  48,  49,  53,  50,  56,  54,  48,  53,  52,  52,  95, 100, 102,  57,  49,  50,  56,  56,  54,  50,  97,  95, 111,  46, 106, 112, 103,   0, 197,  78,  32,   0, 127, 168, 138,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  55,  48,  53,  51,  55,  51,  54,  54,  53,  95,  99, 101,  53,  50,  54,
                      54,  49,  53,  54,  56,  95, 111,  46, 106, 112, 103,   0,  97,  80,  32,   0, 243,  73, 153,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  49,  46,  49,  49,  52,  51,  56,  54,  50,  57,  49,  49,  51,  95,  53,  53,  56,  56,  98,  52,  54,  49,  54, 102,  95, 111,  46, 106, 112, 103,   0, 231,  78,  32,   0,   2,  79, 154,  76,  64,   0,   8, 108,  97, 110,
                     100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  56,  54,  49,  56,  48,  54,  54,  50,  51,  95,  99,  53,  50,  49,  56,  57,  48, 102,  51,  54,  95, 111,  46, 106, 112, 103,   0, 191,  79,  32,   0, 117,  54, 156,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  49,  51,  55,
                      57,  54,  50,  51,  54,  51,  95,  54,  56, 101, 101,  50,  51, 101, 101,  55,  98,  95, 111,  46, 106, 112, 103,   0,  27,  78,  32,   0, 236,   1, 159,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  52,  46,  57,  52,  50,  57,  53,  53,  53,  56,  57,  57,  95,  52,  98,  52,  50,  57, 101,  57, 100,  99,  48,  95, 111,  46, 106, 112, 103,   0,   8, 235,  77,
                      32,   0,  87, 111, 202,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  55,  46,  57,  51,  53,  48,  55,  50,  55,  57,  50,  50,  95,  49,  52,  49,  48,  53, 102,  99,  97, 101,  57,  95, 111,  46, 106, 112, 103,   0,   8, 224,  77,  32,   0, 254, 221, 220,  76,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95,
                     102,  97,  99, 101,  46,  55,  46,  49,  48,  52,  53,  51,  55,  56,  55,  55,  54,  52,  95,  55,  52,  53, 100, 101,  53, 102,  99, 101,  49,  95, 111,  46, 106, 112, 103,   0,  15,  79,  32,   0, 150, 151,  26,  77,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  57,  51,  52,  55,  53,  55,  55,  52,  54,  51,  95,  52, 100,  98,  52, 101,  97,  51, 100,
                     101,  57,  95, 111,  46, 106, 112, 103,   0,   8, 199,  78,  32,   0, 252, 169,  76,  77,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  55,  48,  53,  54,  54,  57,  52,  55,  51,  95,  53,  56,  97,  51, 101,  97,  57, 100, 102,  97,  95, 111,  46, 106, 112, 103,   0, 131,  79,  32,   0,  30, 153, 127,  77,  64,   0,   8, 108,  97, 110, 100, 109,
                      97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  48,  55,  57,  53,  53,  54,  52,  48,  54,  52,  95,  49,  53, 101, 101,  57,  52, 100, 102,  52, 101,  95, 111,  46, 106, 112, 103,   0,  81,  80,  32,   0,  31, 242,   7,  78,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  49,  46,  49,  49,  48,  52,  53,  56,  53,
                      56,  55,  51,  53,  95,  54,  57,  54,  54,  57,  52,  57,  98,  56,  57,  95, 111,  46, 106, 112, 103,   0, 228,  78,  32,   0, 203,  66,  41,  78,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  56,  54,  49,  53,  56,  52,  53,  48,  53,  95, 100, 100,  53,  49,  50,  99,  99,  53,  55,  48,  95, 111,  46, 106, 112, 103,   0, 104,  79,  32,   0,
                     222,  49,  56,  78,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  53,  54,  50,  54,  53,  55,  48,  52,  51,  95,  55,  52,  53,  51, 102,  54,  52, 102,  48,  99,  95, 111,  46, 106, 112, 103,   0,  91,  80,  32,   0, 118,   0,  81,  78,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,
                      99, 101,  46,  49,  46,  49,  49,  52,  48,  57,  50,  54,  48,  57,  55,  54,  95,  56,  97,  56,  48,  54,  57,  51, 101,  51,  56,  95, 111,  46, 106, 112, 103,   0, 236,  78,  32,   0,  13,  56, 146,  78,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  56,  54,  50,  52,  50,  48,  54,  57,  54,  95,  55,  51,  55,  49,  57,  57,  57, 101,  53,
                      53,  95, 111,  46, 106, 112, 103,   0, 107,  78,  32,   0, 245, 115, 152,  78,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  57,  51,  53,  48,  53,  50,  55,  50,  56,  50,  95,  53,  53,  99,  97,  54,  51,  48,  97,  54,  50,  95, 111,  46, 106, 112, 103,   0,   8, 227,  79,  32,   0, 190,  36, 242,  78,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114,
                     107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  48,  50,  48,  52,  52,  49,  56,  49,  48,  52,  95,  56,  52,  56, 102,  50,  99,  50,  54, 102,  53,  95, 111,  46, 106, 112, 103,   0,  52,  78,  32,   0, 233,  14,   0,  79,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  53,  46,  49,  48,  51,  57,  57,  55,  53,  51,  48,
                      49,  54,  95,  49,  48,  97,  98,  53,  53, 101, 101,  54,  50,  95, 111,  46, 106, 112, 103,   0,  90,  79,  32,   0, 196,  41, 141,  79,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  50,  46,  49,  49,  53,  49,  56,  54,  53,  54,  50,  52,  52,  95,  97,  50, 100,  99, 101,  99, 100,  56,  48,  53,  95, 111,  46, 106, 112, 103,   0, 167,  77,  32,   0, 152, 133,
                     152,  79,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101,  46,  57,  46,  49,  48,  52,  53,  57,  50,  51,  49,  54,  49,  53,  95, 102,  51,  51,  97,  99, 100, 100,  55,  56,  56,  95, 111,  46, 106, 112, 103,   0, 137,  78,  32,   0, 211,  20, 182,  79,  64,   0,   8, 108,  97, 110, 100, 109,  97, 114, 107,  95,  97, 108, 105, 103, 110, 101, 100,  95, 102,  97,  99, 101);


                MB_LBP_image( R, p_block, 59, 3, false );
                //MB_LBP_image( iimg, p_block, 59, 3, false );

				        vector<float> features = MultiScaleBlock_Mapping();

				        for (unsigned int i = 0; i < features.size(); ++i) cout << features.at(i) << endl;

                //Mat tmpImg = lbp.getLBPImage();
                //imwrite( "../data/adience/test/lbp.png", tmpImg);

                /* size:
                -hf = 32
                -riu2 = 10
                -ri = 36
                -u2 = 59
                */
                
                // Mat mask( h, w, CV_8UC1 );
                // int n = subi; // divide image to subi*subi blocks
                // vector<double> concat;
                // if (age_label != 27 and gender_label != 2){
                //     train_test++; 
                //     for( int j = 0; j < n; j++ ) {
                //         for( int i = 0; i < n; i++ ) {
                //             // Reset mask. Will actually not allocate the data as it is
                //             // same size as before.
                //             mask = Mat::zeros( h, w, CV_8UC1 );
                //             // Get a sub-image (ROI) the size of 1/4 of the whole image
                //             int x = w / n * i;
                //             int y = h / n * j;
                //             int wH = w / n - n;
                //             int hH = h / n - n;
                //             Mat roi( mask, Range( y, y + hH ), Range( x, x + wH ) );
                //             roi = Scalar( 255 );

                //             vector<double> hist = lbp.calcHist( mask ).getHist(normalizeHist);
                //             for (unsigned int i = 0; i < hist.size(); ++i) concat.push_back(hist.at(i));
                //         }
                //     }
                //     if (train_test==1){
                //         cout << mapping << " nPredictors : "<< concat.size() <<endl;
                //     }
                //     data.push_back(concat);
                //     labels_age.push_back(age_label);
                //     labels_gender.push_back(gender_label);
                // }
                // else{

                // }
                
            }
        }
        else{
            cout << "Error en Dataset, falta alguna columna" << endl;
        }
    }
}

void LBP_dataset::MB_LBP_image(Mat d_img, int p_blocks, int _N_features, int slider, bool borderCopy) {
     
    // Make sure the image has Double precision version
	// if( d_img.type() < CV_64F ) {
	//  	d_img.convertTo( d_img, CV_64F );
	// }
 
    int y =0, x =0;
    h_size = 256;// N_features = 58;
    N_features = _N_features;

    if (N_features<=h_size){
        mb_initialized = true;
      	// Make a copy of the image border the same size as the radius
      	if( borderCopy ) {
      		Mat tmp( d_img.rows+2*p_blocks, d_img.cols+2*p_blocks, CV_64F );
      		copyMakeBorder( d_img, tmp, p_blocks, p_blocks, p_blocks, p_blocks, BORDER_WRAP, Scalar(0) );
      		d_img = tmp.clone();
      	}
      	int xsize = d_img.cols;
      	int ysize = d_img.rows;
      	
      	histogram.resize(h_size,0); // 2 ^ 8 = binary patterns combinations
      	while ((y+ 3*p_block + slider) <= ysize) // ybox = 3 blocks * p_block
      	{
        		while ((x+ 3*p_block + slider) <= xsize) // xbox = 3 blocks * p_block
        		{
                int lbp_code = MultiScale_LBP(d_img, y, x, p_blocks);
          			histogram.at(lbp_code)++;
          			x += slider;
      		}
      		y += slider;
      		x = 0;
    	}  
   }  
   else{
      cout << "Error: N_features should be smaller than 256" << endl;
      mb_initialized = true;
   } 
}

int LBP_dataset::MultiScale_LBP(Mat d_img, int y, int x, int p_blocks){
      int neiboneighborhood = 8;
      int central_rect_y = y + p_blocks; 
      int central_rect_x = x + p_blocks;
      vector<int> y_offsets{-1, -1, -1, 0, 1, 1, 1, 0}; //neiboneighborhood = 8, 8 positions, clockwise direction
      vector<int> x_offsets{-1, 0, 1, 1, 1, 0, -1, -1}; //neiboneighborhood = 8, 8 positions, clockwise direction
      bitset<8> binary_code;  //neiboneighborhood = 8, binary pattern lenght
      int y_shift = p_blocks -1; 
      int x_shift = p_blocks -1;

      for (int i = 0; i < neiboneighborhood; ++i)
      {
          y_offsets.at(i) = p_blocks*y_offsets.at(i);
          x_offsets.at(i) = p_blocks*x_offsets.at(i);
        }

      double central_value = Integrate(d_img, central_rect_y, central_rect_x, central_rect_y +y_shift, central_rect_x + x_shift);
      //cout << central_value << endl;
      for (int i = 0; i < neiboneighborhood; ++i)
      {
          int current_rect_y = central_rect_y + y_offsets.at(i);
          int current_rect_x = central_rect_x + x_offsets.at(i);

          double current_rect_val = Integrate(d_img, current_rect_y, current_rect_x, current_rect_y + y_shift, current_rect_x + x_shift);

          binary_code.set((neiboneighborhood-1) -i, current_rect_val >= central_value);

      }

      return binary_code.to_ulong();
}

vector<float> LBP_dataset::MultiScaleBlock_Mapping(){
  if (!mb_initialized) exit(1);
	sort(histogram.begin(), histogram.end());
  //cout << histogram.size() << endl;
  for (unsigned int i = 0; i < histogram.size(); ++i) cout << histogram.at(i) << endl;		
	vector<float> features(histogram.end()- N_features+1 , histogram.end());
  vector<float> features2(histogram.begin(), histogram.end() - N_features+1);
	features.push_back(accumulate(histogram.begin(), histogram.end() - N_features+1, 0));
  vector<float>::const_iterator max_value; 
  max_value = max_element(features.begin(), features.end());
  transform(features.begin(), features.end(), features.begin(), bind2nd(divides<float>(), *max_value));
	return features;
}

double LBP_dataset::Integrate( Mat d_img, int r0, int c0, int r1, int c1) {
	double S = 0.0;
	S += d_img.at<float>(r1, c1);
	if ((r0-1 >= 0) && (c0 -1 >= 0))
	{
		S+= d_img.at<float>(r0 - 1, c0 - 1);
	}
	if (r0 - 1 >= 0)
	{
		S -= d_img.at<float>(r0 -1, c1);
	}
	if (c0 -1 >= 0)
	{
		S -= d_img.at<float>(r1, c0 -1);
	}
	return S;
}

void LBP_dataset::create_in_file(string _output, string _type) {
    if (!initialized) exit(1);
    output_path= _output;

    ofstream ofs_dataset;// files output_path
    ofstream ofs_age_label;
    ofstream ofs_gender_label;
    ofs_dataset.open(output_path+"/dataset.csv", ios::out );

    if (_type == "age")
    {

        ofs_age_label.open( output_path+"/age_label.csv", ios::out );
    }
    else {
        
        ofs_gender_label.open( output_path+"/gender_label.csv", ios::out );
    }
    
    if(_type != "gender" &&  _type != "age") cout << "Wrong Type, Default: gender" << endl;
    

    int rows = data.size();
    int cols = data.at(0).size();
    for (int i = 0; i < rows; ++i){
        int j;
        for (j = 0; j < cols-1; ++j){
            ofs_dataset << (data.at(i)).at(j) << ",";
        }
        ofs_dataset << (data.at(i)).at(j) << endl;
    }

    if (_type == "age"){
        int rows = labels_age.size();
        for (int i = 0; i < rows; ++i) ofs_age_label << labels_age.at(i) << endl;
    }
    else {
        int rows = labels_gender.size();
        for (int i = 0; i < rows; ++i) ofs_gender_label << labels_gender.at(i) << endl;
    }
    
    ofs_dataset.close();
    ofs_age_label.close();
    ofs_gender_label.close();    
    
}

void LBP_dataset::get_dataset(MatrixXd &_data, VectorXi &_labels, string _type) {
    if (!initialized) exit(1);
    int rows = data.size();
    int cols = data.at(0).size();
    _data.resize(rows,cols);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) _data(i,j) =(data.at(i)).at(j);

    if (_type == "age"){
        int rows = labels_age.size();
        _labels.resize(rows);
        for (int i = 0; i < rows; ++i) _labels(i) =labels_age.at(i);
    }
    else {
        int rows = labels_gender.size();
        _labels.resize(rows);
        for (int i = 0; i < rows; ++i) _labels(i) =labels_gender.at(i);
    }

    if(_type != "gender" &&  _type != "age") cout << "Wrong Type, Default: gender" << endl;
    
}



int LBP_dataset::translate(string dictionary[], string data_class, string target){
    int size;
    if (data_class == "gender"){
        size = n_gender;
        for (int i = 0; i < size; ++i) if (target == dictionary[i]) return i;
    }
    else {
        size = n_age;
        for (int i = 0; i < size; ++i)
        {
            if (target == dictionary[i])
            {
                switch(i)
                {
    		      case 0: return 0;
        		  case 1: return 0;
        		  case 2: return 0;
        		  case 3: return 1;
        		  case 4: return 1;
        		  case 5: return 2;
        		  case 6: return 2;
        		  case 7: return 3;
        		  case 8: return 3;
        		  case 9: return 3;
        		  case 10: return 4;
        		  case 11: return 5;
        		  case 12: return 1;
        		  case 13: return 0;
        		  case 14: return 1;
        		  case 15: return 1;
        		  case 16: return 2;
        		  case 17: return 0;
        		  case 18: return 3;
        		  case 19: return 3;
        		  case 20: return 3;
        		  case 21: return 3;
        		  case 22: return 3;
        		  case 23: return 3;
        		  case 24: return 4;
        		  case 25: return 4;
        		  case 26: return 4;
        		  case 27: return 27;
    		  }
            }
        }
    }
    return 27;


}


