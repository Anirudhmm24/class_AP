// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include "SRHT.h"

// [[Rcpp::export]]
Rcpp::List SRHT_cpp(Eigen::MatrixXd X, Eigen::VectorXd y, int r, Rcpp::Nullable<int> seed = R_NilValue)
{
    std::optional<uint64_t> seed_val = std::nullopt;
    if (seed.isNotNull()) {
        seed_val = static_cast<uint64_t>(Rcpp::as<int>(seed));
    }
    auto result = SRHT(X, y, r, seed_val);
    return Rcpp::List::create(
        Rcpp::Named("X_f") = result.first,
        Rcpp::Named("y_f") = result.second
    );
}
