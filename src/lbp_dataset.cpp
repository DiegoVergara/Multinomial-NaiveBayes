
#include "../include/lbp_dataset.hpp"

LBP_dataset::LBP_dataset()
{
    initialized=false;
}

LBP_dataset::LBP_dataset(string _image_path, string _fn_csv, int _rad, int _pts, int _subi, string _mapping, bool _normalizeHist)
{   
    image_path = _image_path;
    fn_csv = _fn_csv;
    //output_path = _output_path;
    rad = _rad;
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
    int train_test=0;
    //while (getline(csv, line) && i<max) {
    while (getline(csv, line)) {
        stringstream liness(line);
        path.clear(); age.clear(); gender.clear();
        getline(liness, path, separator);
        getline(liness, age, separator); // age
        getline(liness, gender, separator); // gender
        if(!path.empty() && !age.empty() && !gender.empty()) { 

            
            int age_label = translate(v_age, "age", age.c_str());
            int gender_label = translate(v_gender, "gender", gender.c_str());
            
            // 'path' can be file, dir or wildcard path

            String root((image_path+path).c_str());
            vector<String> files;
            glob(root, files, true);
            for(vector<String>::const_iterator f = files.begin(); f != files.end(); ++f) {
                Mat img = imread(*f, IMREAD_GRAYSCALE); // lead image to mat object in grayscale

                Size size(240, 240);
                resize(img,img,size); //scale image to size

                int w = img.cols, h = img.rows;

                static bool showSmallSizeWarning = true;
                if(w>0 && h>0 && (w!=img.cols || h!=img.rows)) cout << "\t* Warning: images should be of the same size!" << endl;
                if(showSmallSizeWarning && (img.cols<50 || img.rows<50)) {
                    cout << "* Warning: for better results images should be not smaller than 50x50!" << endl;
                    showSmallSizeWarning = false;
                }
                equalizeHist(img, img); //Equalize Image
                img.convertTo( img, CV_64F );

                //Mat tmpImg = lbp.getLBPImage();
                //imwrite( "lbp.png", img);

                LBP lbp( pts, LBP::strToType( mapping ) );

                lbp.calcLBP( img, rad, true );

                /* size:
                -hf = 32
                -riu2 = 10
                -ri = 36
                -u2 = 59
                */
                
                Mat mask( h, w, CV_8UC1 );
                int n = subi; // divide image to subi*subi blocks
                vector<double> concat;
                if (age_label != 27 and gender_label != 2){
                    train_test++; 
                    for( int j = 0; j < n; j++ ) {
                        for( int i = 0; i < n; i++ ) {
                            // Reset mask. Will actually not allocate the data as it is
                            // same size as before.
                            mask = Mat::zeros( h, w, CV_8UC1 );
                            // Get a sub-image (ROI) the size of 1/4 of the whole image
                            int x = w / n * i;
                            int y = h / n * j;
                            int wH = w / n - n;
                            int hH = h / n - n;
                            Mat roi( mask, Range( y, y + hH ), Range( x, x + wH ) );
                            roi = Scalar( 255 );

                            vector<double> hist = lbp.calcHist( mask ).getHist(normalizeHist);
                            for (unsigned int i = 0; i < hist.size(); ++i) concat.push_back(hist.at(i));
                        }
                    }
                    if (train_test==1){
                        cout << mapping << " nPredictors : "<< concat.size() <<endl;
                    }
                    data.push_back(concat);
                    labels_age.push_back(age_label);
                    labels_gender.push_back(gender_label);
                }
                else{

                }
                
            }
        }
        else{
            cout << "Error en Dataset, falta alguna columna" << endl;
        }
    }
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


