#include <iostream>
#include <iomanip>  
#include <Eigen/Dense>
#include <Eigen/Core>
#include "../include/multinomialnaivebayes.hpp"
#include "../include/c_utils.hpp"
#include <string>

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{

  string data_csv_path, labels_csv_path;
  int rows, cols, train_partition;
  double alpha;
  
  if(argc == 5){
    data_csv_path = argv[1];
    labels_csv_path = argv[2];
    alpha = atof(argv[3]);
    train_partition = atoi(argv[4]);
  }
  else{
    cout << "Arguments :" << argc-1 << "/4" << endl;
    cout << "\nRun: ./test_gaussian [options]" << endl;
    cout << "\nOptions:" << endl;
    cout << "\t<string> path of data csv file" << endl;
    cout << "\t<string> path of labels csv file'" << endl;
    cout << "\t<double> alpha" << endl;
    cout << "\t<int> Number of rows data train\n" << endl;
    exit(1);
  }
/*
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
  VectorXi predicted_labels;

  int n_data = data_train.rows();
  //VectorXd w = VectorXd::Ones(n_data)/n_data;
  VectorXd w = VectorXd::Ones(n_data);
*/
  VectorXd w = VectorXd::Ones(8)/ 8;
  
  MatrixXd data_train(8,3),data_test(3,3);
  VectorXi labels_train(8),labels_test(3);
  
  data_train << 6,180,6,
          5.92,190,8,
          5.58,170,9,
          5.92,165,7,
          5,100,9,
          5.5,150,8,
          5.42,130,9,
          5.75,150,8;

  data_test << 6,130,8,
            5.92,190,8,
            5.58,170,9;

  labels_train << 1,
            1,
            1,
            1,
            0,
            0,
            0,
            0;

  labels_test << 0,
            1,
            1;

  MultinomialNaiveBayes naive_bayes(data_train,labels_train);
  naive_bayes.fit(alpha, w);
  //predicted_labels=naive_bayes.test(data_test);
  cout << naive_bayes.get_proba(data_test) << endl;
  //utils.classification_Report(labels_test, predicted_labels);
  return 0;
}

