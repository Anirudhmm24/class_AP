#pragma once

#include <Eigen/Dense>
#include <utility>

std::pair<Eigen::MatrixXd, Eigen::VectorXd> HT(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
