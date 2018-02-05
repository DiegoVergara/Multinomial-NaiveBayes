#include "../include/gaussiannaivebayes_col.hpp"

GaussianNaiveBayes::GaussianNaiveBayes()
{
    initialized=false;
}

GaussianNaiveBayes::GaussianNaiveBayes(MatrixXd &datos,VectorXi &clases)
{
    X=&datos;
    Y=&clases;
    initialized=true;
}
void GaussianNaiveBayes::fit(VectorXd weights)
{   
    if (initialized){
        int rows = X->rows();
        int cols = X->cols();

        for (int i = 0; i < rows; ++i) {        
            if(means[(*getY())(i)].size()==0){
                means[(*getY())(i)] = VectorXd::Zero(cols);
                ix[(*getY())(i)] = 0;
                ac_weight[(*getY())(i)] = 0;
            }
            means[(*getY())(i)] += weights(i)*X->row(i);
            ix[(*getY())(i)] += 1;
            ac_weight[(*getY())(i)]+= weights(i);
        }

        for (int i = 0; i < rows; ++i){   
            if(sigmas[(*getY())(i)].size()==0){
                means[(*getY())(i)] /= ac_weight[(*getY())(i)];
                sigmas[(*getY())(i)] = VectorXd::Zero(cols);
            }
            sigmas[(*getY())(i)] =  sigmas[(*getY())(i)].array() +((X->row(i).transpose() - means[(*getY())(i)]).array().square() * weights(i)); 
        }  

        std::map<unsigned int,int>::iterator iter;
        for (iter = ix.begin(); iter != ix.end(); ++iter) {             
            sigmas[iter->first] /= ac_weight[iter->first];
            Prior[iter->first] = (iter->second + 0.0)/rows;
        }
    }
}

// double GaussianNaiveBayes::log_likelihood(VectorXd data, VectorXd mean, VectorXd sigma){
//     double loglike =0.0;
//     loglike = -0.5 * ((2*M_PI*sigma).array().log()).sum();
//     loglike -= 0.5 * (((data - mean).array().square())/sigma.array()).sum();
//     return loglike;
// }


VectorXd GaussianNaiveBayes::log_likelihood(VectorXd data, VectorXd mean, VectorXd sigma){
    int cols = data.rows();
    VectorXd loglike = VectorXd::Zero(cols);
    for (int i = 0; i < cols; ++i)
    {
        loglike(i) = -0.5 * (log(2*M_PI*sigma(i)));
        loglike(i) -= 0.5 * ((pow(data(i) - mean(i), 2))/sigma(i));
    }

    return loglike;
}

// VectorXd GaussianNaiveBayes::test(MatrixXd &Xtest)
// {
//     VectorXd c=VectorXd::Zero(Xtest.rows());
//     if (initialized){
//         double max_class=0.0;
//         double max_score=-100000000.0;
//         double score=0;
//         std::map<unsigned int,int>::iterator iter;
//         #pragma omp parallel for private(max_class,max_score,score,iter)
//         for (int i = 0; i < Xtest.rows(); ++i) {
//             max_class=0.0;
//             max_score= -100000000.0;
//             for (iter = ix.begin(); iter != ix.end(); ++iter) {  
//                 score=log(getPrior()[iter->first]) + log_likelihood(Xtest.row(i), means[iter->first], sigmas[iter->first]);
//                 if(score > max_score){
//                     max_score=score;
//                     max_class=iter->first;
//                 }
//             }
//             c(i)=max_class;
//         }
//         return c;
//     }
//     else{
//         return c;
//     }

// }

MatrixXd GaussianNaiveBayes::test(MatrixXd &Xtest)
{
    MatrixXd c(Xtest.rows(), Xtest.cols());
    if (initialized){
        VectorXd score;
        std::map<unsigned int,int>::iterator iter;
        //#pragma omp parallel for private(max_class,max_score,score,iter)
        for (int i = 0; i < Xtest.rows(); ++i) {
            VectorXd max_class = VectorXd::Zero(Xtest.cols());
            VectorXd max_score = VectorXd::Ones(Xtest.cols()) * -100000000.0;
            for (iter = ix.begin(); iter != ix.end(); ++iter) {  
                score=log_likelihood(Xtest.row(i), means[iter->first], sigmas[iter->first]).array() + log(getPrior()[iter->first]);
                for (int j = 0; j < Xtest.cols(); ++j){
                    if(score(j) > max_score(j)){
                        max_score(j)=score(j);
                        max_class(j)=iter->first;
                    }
                }
            }
            c.row(i)=max_class;
        }
        return c;
    }
    else{
        return c;
    }
}


std::map<unsigned int, double> GaussianNaiveBayes::getPrior() const
{
    return Prior;
}

void GaussianNaiveBayes::setPrior(const std::map<unsigned int, double> &value)
{
    Prior = value;
}
 MatrixXd *GaussianNaiveBayes::getX() 
{
    return X;
}

void GaussianNaiveBayes::setX( MatrixXd *value)
{
    X = value;
}
 VectorXi *GaussianNaiveBayes::getY() 
{
    return Y;
}

void GaussianNaiveBayes::setY( VectorXi *value)
{
    Y = value;
}

