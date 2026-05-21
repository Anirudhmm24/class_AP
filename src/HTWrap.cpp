// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include "HT.hpp"

// [[Rcpp::export]]
Rcpp::List HT_cpp(Eigen::MatrixXd X, Eigen::VectorXd y)
{
    auto result = HT(X, y);
    return Rcpp::List::create(
    Rcpp::Named("X_f") = result.first,
    Rcpp::Named("y_f") = result.second
  );
}
