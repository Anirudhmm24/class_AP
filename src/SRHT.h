#pragma once

#include <Eigen/Dense>
#include <utility>
#include <cstdint>
#include <optional>

std::pair<Eigen::MatrixXd, Eigen::VectorXd> SRHT(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const int &r,
    std::optional<uint64_t> seed = std::nullopt
);
