#include "../include/lbp_dataset.hpp"
#include "../include/adaboost.hpp"
#include "../include/c_utils.hpp"

using namespace std;
using namespace Eigen;


int main(int argc, char const *argv[])
{
	string image_path, fn_csv, mapping, type, algorithm;
	int rad, pts, subi, n_estimators, n; 
	bool normalizeHist;
	double alpha, learning_rate;
	
	if(argc == 14){
		image_path = argv[1];
	   	fn_csv = argv[2];
	   	type = argv[3];
	   	mapping = argv[4];
	   	stringstream ss(argv[5]);
	   	if(!(ss >> boolalpha >> normalizeHist)) {
    		cout << "Error en valor booleano: normalizeHist" << endl;
    		exit(1);
		}
		rad = atoi(argv[6]);
		pts = atoi(argv[7]);
		subi = atoi(argv[8]);

		n_estimators = atoi(argv[9]);
		alpha = atof(argv[10]);
		learning_rate= atof(argv[11]);
		algorithm = argv[12];
		n = atoi(argv[13]);

	}
	else{
		cout << "Arguments :" << argc-1 << "/13" << endl;
	    cout << "\nRun: ./test_dataset [options]" << endl;
	    cout << "\nOptions:" << endl;
	    cout << "\tLBP:" << endl;
	    cout << "\t<string> path of image folder" << endl;
	    cout << "\t<string> path of 'dataset.txt'" << endl;
	    cout << "\t<string> Type: 'age' or 'gender'" << endl;
		cout << "\t<string> Mapping choose between:" << endl;
		cout << "\t\tu2\n" << "\t\tri\n" << "\t\triu2\n" << "\t\thf" << endl;
		cout << "\t<bool> Output normalized histogram instead of LBP image: 'true' or 'false'" << endl;
		cout << "\t<int> Radius" << endl;
		cout << "\t<int> Number of support points" << endl;
		cout << "\t<int> Number of image blocks" << endl;
		cout << "\tADABOOST:" << endl;
		cout << "\t<int> Number of estimators" << endl;
		cout << "\t<double> alpha - Multinomial algorithm, any for Gaussian" << endl;
		cout << "\t<double> learning rate" << endl;
		cout << "\t<string> Adaboost algorithm:" << endl;
		cout << "\t\tsamme\n" << "\t\tsamme.r" << endl;
		cout << "\t<int> Number of row data train\n" << endl;


		exit(1);
	}

	MatrixXd data;
	VectorXi labels, predicted_labels;

    cout << "Init Dataset" << endl;
	LBP_dataset dataset(image_path, fn_csv, rad, pts, subi, mapping, normalizeHist);
	cout << "Save Dataset" << endl;
	dataset.get_dataset(data, labels, type);


	MatrixXd data_train =  data.block(0,0 ,n, data.cols());
	MatrixXd data_test =  data.block(n,0 ,data.rows()-n, data.cols());
	VectorXi labels_train =  labels.head(n);
	VectorXi labels_test =  labels.tail(labels.rows()-n);

	cout << "Init Adaboost" << endl;
	Adaboost ensemble(algorithm, n_estimators, alpha, learning_rate);
  	cout << "Fit Adaboost" << endl;
	ensemble.fit(data_train, labels_train);
  	cout << "Predict" << endl;
  	predicted_labels = ensemble.predict(data_test);
  	C_utils utils;
  	utils.print(labels_test, predicted_labels);



	return 0;
}