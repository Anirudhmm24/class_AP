#include "sublime_control.h"
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include "HT.h"

// [[Rcpp::export]]
Rcpp::List HT_cpp(Eigen::MatrixXd X, Eigen::VectorXd y, SEXP seed = R_NilValue)
{
    sublime_data_check(X, y);
    auto result = HT(X, y, sublime_seed(seed));
    return Rcpp::List::create(
    Rcpp::Named("X_f") = result.first,
    Rcpp::Named("y_f") = result.second
  );
}
