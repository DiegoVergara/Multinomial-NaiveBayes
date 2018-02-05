#include "../include/adaboost_col.hpp"
#include "../include/c_utils.hpp"

Adaboost::Adaboost()
{
    initialized=false;
}

Adaboost::Adaboost(int _n_estimators, double _alpha, double _learning_rate)
{

    n_estimators = _n_estimators;
    //alpha = _alpha;
    learning_rate = _learning_rate;


    initialized=true;
}


//double Adaboost::boost(VectorXd &w, int iteration, VectorXd &errors, vector<GaussianNaiveBayes> &classifiers){
double Adaboost::boost(VectorXd &w, int iteration, VectorXd &errors, vector<GaussianNaiveBayes> &classifiers, VectorXi &columns){
  

  //VectorXi predicted_labels;
  MatrixXi predicted_labels; 
  VectorXd index = VectorXd::Zero(n_data);
  ///////////////////////////////
  //MultinomialNaiveBayes naive_bayes(*getdata(),*getlabels());
  //naive_bayes.fit(alpha);
  //predicted_labels=naive_bayes.test(*getdata());
  GaussianNaiveBayes naive_bayes(*getdata(),*getlabels());
  naive_bayes.fit(w);
  predicted_labels=naive_bayes.test(*getdata());
  ///////////////////////////////

  // for (int i = 0; i < predicted_labels.rows(); ++i) index(i) = ((predicted_labels(i,k) == (*getlabels())(i)) ? 0: 1);//
  // double e = (w.cwiseProduct(index)).colwise().sum()*(w.colwise().sum()).inverse(); 

  double e, e_menor = 100000.0; 
  int col_menor; //
  for (int k = 0; k < predicted_labels.cols(); ++k){ 
    for (int i = 0; i < predicted_labels.rows(); ++i) index(i) = ((predicted_labels(i,k) == (*getlabels())(i)) ? 0: 1);//
    e = (w.cwiseProduct(index)).colwise().sum()*(w.colwise().sum()).inverse(); 
    if (e< e_menor){
      e_menor = e;
      col_menor = k;
    }
  }
  e = e_menor;
  columns(iteration) = col_menor;
  //--------

  if (e <= 0.0){
    cout << "e negativo" << endl;
    w = VectorXd::Zero(n_data);
    errors(iteration) = 0.0;
    return -1.0;
  }  

  if (e >= 1.0 - (1.0 / n_classes)){
    if (iteration == 0) cout << "Critico: Error en primera Iteracion" << endl;
    cout << "e muy grande" << endl;
    w = VectorXd::Zero(n_data);
    errors(iteration) = 0.0;
    return 0.0;
  }

  double alpha = learning_rate * (log((1.0-e) / e) + log(n_classes-1.0)); 
  if (iteration != (n_estimators-1)) w = w.array() * ((index*alpha).array().exp());

  classifiers.push_back(naive_bayes);
  errors(iteration) = e;
  
  return alpha;
}


void Adaboost::fit(MatrixXd &data, VectorXi &labels){
  dX=&data;
  dY=&test;
  dim = getdata()->cols();
  n_data = getdata()->rows();
  C_utils utils;
  classes  = utils.get_Classes(*(&labels));
  n_classes = classes.size();

  if (initialized){

    VectorXd w = VectorXd::Ones(n_data)/ n_data;
    alphas = VectorXd::Zero(n_estimators);
    VectorXd errors = VectorXd::Ones(n_estimators);

    columns = VectorXi::Zero(n_estimators);//

    //vector<GaussianNaiveBayes> classifiers;
  	initialized=true;

    for (int i = 0; i < n_estimators; ++i)
    {
      cout << i << endl;   
      //alphas(i) = boost(w, i, errors, classifiers);
      alphas(i) = boost(w, i, errors, classifiers, columns);//
      double w_sum = w.sum();

      if ((w_sum <= 0) or ( errors(i) == 0.0))
      {
        cout << "Break" << endl;
        n_estimators = i;
        break;
      }

      if (i < (n_estimators-1)) w /= w_sum;
      
    }
  }
}

 VectorXi Adaboost::predict( MatrixXd &test){
    lX=&labels;
    n_data_test = gettest()->rows();
    // Predict 
    cout <<"predict"<< endl;

    MatrixXd pred = MatrixXd::Zero(n_data_test,n_classes);
    VectorXi result = VectorXi::Zero(n_data_test);
    MatrixXi temp_Yhat = MatrixXi::Zero(n_data_test,1);

    // MatrixXd pred = MatrixXd::Zero(n_data_test,n_classes);
    // VectorXi result = VectorXi::Zero(n_data_test);
    // VectorXd temp_Yhat;

    for (int i = 0; i < n_estimators; ++i)
    {
      temp_Yhat = gettest()->col(columns(i)); //
      temp_Yhat= classifiers.at(i).test(temp_Yhat);//
      //temp_Yhat= classifiers.at(i).test(*gettest());
      //for (int j =0; j< n_classes; ++j) for (int k = 0; k < temp_Yhat.rows(); ++k) pred(k,j) += ((temp_Yhat(k) != classes->at(j)) ? 1: 0)*alphas(i);
      for (int j =0; j< n_classes; ++j) for (int k = 0; k < temp_Yhat.rows(); ++k) pred(k,j) += ((temp_Yhat(k,0) != classes.at(j)) ? 1: 0)*alphas(i);//
    }
    pred /= alphas.sum();
    if (n_classes == 2)
    {
      pred.col(0) *= -1;
      pred.col(0) += pred.col(1);
      for (int j =0; j< pred.rows(); ++j) result(j) = ((pred(j,0) > 0) ? classes.at(0): classes.at(1));
    }
    else{
      MatrixXf::Index   maxIndex[pred.rows()];
      for (int j =0; j< pred.rows(); ++j){
        pred.row(j).maxCoeff(&maxIndex[j]);
        result(j) = maxIndex[j];
       }
    }
    return result;
}


 MatrixXd *Adaboost::getdata() 
{
    return dX;
}

 MatrixXd *Adaboost::gettest() 
{
    return dY;
}

 VectorXi *Adaboost::getlabels() 
{
    return lX;
}
