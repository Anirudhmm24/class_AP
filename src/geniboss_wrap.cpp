// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "geniboss.h"

// [[Rcpp::export]]
Rcpp::List geniboss_cpp(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const Eigen::VectorXd &wt, int r) {

    Eigen::MatrixXd X_selected;
    Eigen::VectorXd y_selected;

    geniboss(X, y, wt, r, X_selected, y_selected);
    return Rcpp::List::create(
        Rcpp::Named("X_selected") = X_selected,
        Rcpp::Named("y_selected") = y_selected
    );
}
