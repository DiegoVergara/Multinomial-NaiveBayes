#include "../include/lbp_dataset.hpp"
#include <sstream>

using namespace std;
using namespace Eigen;

int main(int argc, char const *argv[])
{
	string image_path, fn_csv, mapping, type, output;
	int rad, pts, subi;
	bool normalizeHist;
	
	if(argc == 10){
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
		output = argv[9];
	}
	else{
		cout << "Argumentos :" << argc-1 << "/9" << endl;
	    cout << "\nRun: ./test_dataset [options]" << endl;
	    cout << "\nOptions:" << endl;
	    cout << "\t<string> path of image folder" << endl;
	    cout << "\t<string> path of 'dataset.txt'" << endl;
	    cout << "\t<string> Type: 'age' or 'gender'" << endl;

		cout << "\t<string> Mapping choose between:" << endl;
		cout << "\t\tu2\n" << "\t\tri\n" << "\t\triu2\n" << "\t\thf" << endl;
		cout << "\t<bool> Output normalized histogram instead of LBP image: 'true' or 'false'" << endl;
		cout << "\t<int> Radius" << endl;
		cout << "\t<int> Number of support points" << endl;
		cout << "\t<int> Number of image blocks" << endl;
		cout << "\t<string> Output path\n" << endl;

		exit(1);
	}

	MatrixXd data_train,data_test;
	VectorXi labels_train,labels_test;

	LBP_dataset dataset(image_path, fn_csv, rad, pts, subi, mapping, normalizeHist);
	//dataset.get_dataset(data_train, labels_train, type);
	dataset.create_in_file(output, type);
	return 0;

}