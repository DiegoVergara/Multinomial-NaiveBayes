#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
#include <Eigen/Core>
#include "../include/gaussiannaivebayes.hpp"
#include "../include/c_utils.hpp"
#include <string>

using namespace std;
using namespace Eigen;

 int main(int argc, char *argv[])
{

  string data_csv_path, labels_csv_path;
  int rows, cols, train_partition;
  
  if(argc == 4){
    data_csv_path = argv[1];
    labels_csv_path = argv[2];
    train_partition = atoi(argv[3]);
  }
  else{
    cout << "Arguments :" << argc-1 << "/3" << endl;
    cout << "\nRun: ./test_gaussian [options]" << endl;
    cout << "\nOptions:" << endl;
    cout << "\t<string> path of data csv file" << endl;
    cout << "\t<string> path of labels csv file'" << endl;
    cout << "\t<int> Number of rows data train\n" << endl;
    exit(1);
  }

  MatrixXd data;
  VectorXi labels;

  C_utils utils;

  rows = utils.get_Rows(labels_csv_path);
  cols = utils.get_Cols(data_csv_path, ',');

  utils.read_Data(data_csv_path,data,rows,cols);
  utils.read_Labels(labels_csv_path,labels,rows);

  MatrixXd data_train =  data.block(0,0 ,train_partition, data.cols());
  MatrixXd data_test =  data.block(train_partition,0 ,data.rows()-train_partition, data.cols());
  VectorXi labels_train =  labels.head(train_partition);
  VectorXi labels_test =  labels.tail(labels.rows()-train_partition);
  //VectorXi predicted_labels;

  int n_data = data_train.rows();

  VectorXd w = VectorXd::Ones(n_data)/n_data;

  GaussianNaiveBayes naive_bayes(data_train,labels_train);
  naive_bayes.fit(w);
  //predicted_labels=naive_bayes.test(data_test);
  MatrixXd  predict_proba;
  predict_proba = naive_bayes.get_proba(data_test);
  cout << predict_proba << endl;

  //utils.classification_Report(labels_test, predicted_labels);

  return 0;
}

/*

  
  VectorXd w = VectorXd::Ones(8)/ 8;
  
  MatrixXd data_train(8,2),data_test(1,3);
  VectorXi labels_train(8),labels_test(1);
  
  data_train << 6,180,
          5.92,190,
          5.58,170,
          5.92,165,
          5,100,
          5.5,150,
          5.42,130,
          5.75,150;

  data_test << 6,130,8;

  labels_train << 1,
            1,
            1,
            1,
            0,
            0,
            0,
            0;

  labels_test << 0;

  int n_data = 8;
  int n_classes = 2;
  MatrixXd y_coding = MatrixXd::Ones(n_data, n_classes)*(-1. / (n_classes -1.));
  //cout << y_coding.array().cwiseProduct(data_train.array()).rowwise().sum() << endl;
  cout << y_coding << endl;
  cout << data_train<< endl;
  VectorXd temp(data_train.rows());
  temp = (1. / n_classes) * data_train.rowwise().sum();
  for (int i = 0; i < data_train.rows(); ++i) for (int j = 0; j < data_train.cols(); ++j) data_train(i,j) -= temp(i);
  cout <<  (n_classes - 1) * data_train << endl;
*/
