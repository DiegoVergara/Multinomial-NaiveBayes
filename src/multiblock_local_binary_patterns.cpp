#include "../include/multiblock_local_binary_patterns.hpp"

MSB_LocalBinaryPatterns::MSB_LocalBinaryPatterns()
{
    initialized=false;
}

MSB_LocalBinaryPatterns::MSB_LocalBinaryPatterns(int _p_block, int _n_features, int _slider, bool _copy_border, bool _multiscale){   
	p_blocks = _p_block;
	n_features = _n_features;
	slider = _slider;
	copy_border = _copy_border;
	multiscale = _multiscale;
	initialized=true;
}

void MSB_LocalBinaryPatterns::getFeatureValue(Mat& _image, vector<Rect> _sampleBox, bool _isPositiveBox){
	if (!initialized) exit(1);
	int xMin, xMax, yMin, yMax;
	for (unsigned int k = 0; k < _sampleBox.size(); ++k){
		Rect box = _sampleBox.at(k);

		xMin = MIN(MAX(box.x,0),_image.cols);
		xMax = MIN(box.x + box.width, _image.cols);
		yMin = MIN(MAX(box.y,0),_image.rows);
		yMax = MIN(box.y + box.height, _image.rows);

		Mat auxSubImage = _image(Rect(xMin, yMin, xMax-xMin, yMax-yMin));
		Mat subImage;
		auxSubImage.copyTo(subImage);
		
		if (!multiscale){
			Size size(64,128);
			resize(subImage, subImage, size);
		}

		multiScaleBlock_Image( subImage );
		vector<float> hist = multiScaleBlock_Mapping();
		//for (unsigned int i = 0; i < hist.size(); ++i) cout << hist.at(i) << endl;		

        if(_isPositiveBox){
	        for (unsigned int i = 0; i < hist.size(); ++i){
	        	sampleFeatureValue(k,i) = hist[i];
	        }
	    }else{
	    	for (unsigned int i = 0; i < hist.size(); ++i){
	        	negativeFeatureValue(k,i) = hist[i];
	        }
	    }
	}
}

void MSB_LocalBinaryPatterns::init(Mat& _image, vector<Rect> _sampleBox){
	if (!initialized) exit(1);
	sampleFeatureValue = MatrixXd(_sampleBox.size(),n_features);
    negativeFeatureValue = MatrixXd(_sampleBox.size(),n_features);
    getFeatureValue(_image, _sampleBox, true);
}

void MSB_LocalBinaryPatterns::multiScaleBlock_Image(Mat& d_img) {
     
     if (!initialized) exit(1);
    // Make sure the image has Double precision version
	// if( d_img.type() < CV_64F ) {
	//  	d_img.convertTo( d_img, CV_64F );
	// }
 
    int y =0, x =0;
    h_size = 256;

    if (n_features<=h_size){

        initialized = true;
      	// Make a copy of the image border the same size as the radius
      	if( copy_border ) {
      		Mat tmp( d_img.rows+2*p_blocks, d_img.cols+2*p_blocks, CV_64F );
      		copyMakeBorder( d_img, tmp, p_blocks, p_blocks, p_blocks, p_blocks, BORDER_WRAP, Scalar(0) );
      		d_img = tmp.clone();
      	}
      	int xsize = d_img.cols;
      	int ysize = d_img.rows;
      	
      	histogram.resize(h_size,0); // 2 ^ 8 = binary patterns combinations
      	while ((y+ 3*p_blocks + slider) <= ysize){ // ybox = 3 blocks * p_block
        		while ((x+ 3*p_blocks + slider) <= xsize){  // xbox = 3 blocks * p_block
                	int lbp_code = multiScaleBlock_LBP(d_img, y, x);
          			histogram.at(lbp_code)++;
          			x += slider;
      		}
      		y += slider;
      		x = 0;
    	}  
   }  
   else{
      cout << "Error: N_features should be smaller than 256" << endl;
      initialized = false;
   } 
}

int MSB_LocalBinaryPatterns::multiScaleBlock_LBP(Mat& d_img, int y, int x){
	if (!initialized) exit(1);
	int neiboneighborhood = 8;
	int central_rect_y = y + p_blocks; 
	int central_rect_x = x + p_blocks;
	vector<int> y_offsets{-1, -1, -1, 0, 1, 1, 1, 0}; //neiboneighborhood = 8, 8 positions, clockwise direction
	vector<int> x_offsets{-1, 0, 1, 1, 1, 0, -1, -1}; //neiboneighborhood = 8, 8 positions, clockwise direction
	bitset<8> binary_code;  //neiboneighborhood = 8, binary pattern lenght
	int y_shift = p_blocks -1; 
	int x_shift = p_blocks -1;

	for (int i = 0; i < neiboneighborhood; ++i){
	  y_offsets.at(i) = p_blocks*y_offsets.at(i);
	  x_offsets.at(i) = p_blocks*x_offsets.at(i);
	}

	double central_value = Integrate(d_img, central_rect_y, central_rect_x, central_rect_y +y_shift, central_rect_x + x_shift);

	for (int i = 0; i < neiboneighborhood; ++i){
	  int current_rect_y = central_rect_y + y_offsets.at(i);
	  int current_rect_x = central_rect_x + x_offsets.at(i);
	  double current_rect_val = Integrate(d_img, current_rect_y, current_rect_x, current_rect_y + y_shift, current_rect_x + x_shift);
	  binary_code.set((neiboneighborhood-1) -i, current_rect_val >= central_value);
	}
  	return binary_code.to_ulong();
}

double MSB_LocalBinaryPatterns::Integrate(Mat& d_img, int r0, int c0, int r1, int c1) {
	double S = 0.0;
	S += d_img.at<float>(r1, c1);
	if ((r0-1 >= 0) && (c0 -1 >= 0)){
		S+= d_img.at<float>(r0 - 1, c0 - 1);
	}
	if (r0 - 1 >= 0){
		S -= d_img.at<float>(r0 -1, c1);
	}
	if (c0 -1 >= 0){
		S -= d_img.at<float>(r1, c0 -1);
	}
	return S;
}

vector<float> MSB_LocalBinaryPatterns::multiScaleBlock_Mapping(){
	if (!initialized) exit(1);
	sort(histogram.begin(), histogram.end());
	//for (unsigned int i = 0; i < histogram.size(); ++i) cout << histogram.at(i) << endl;		
	vector<float> features(histogram.end()- n_features+1 , histogram.end());
	features.push_back(accumulate(histogram.begin(), histogram.end() - n_features+1, 0));
	vector<float>::const_iterator max_value; 
	max_value = max_element(features.begin(), features.end());
	transform(features.begin(), features.end(), features.begin(), bind2nd(divides<float>(), *max_value));
	return features;
}
