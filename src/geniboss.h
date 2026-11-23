#pragma once
#include <RcppEigen.h>

void geniboss(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &wt,
    int c,
    Eigen::MatrixXd &X_selected,
    Eigen::VectorXd &y_selected
);
