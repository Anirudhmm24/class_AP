#pragma once

#include <Eigen/Dense>
#include <utility>
#include <optional>
#include <cstdint>

std::pair<Eigen::MatrixXd, Eigen::VectorXd> HT(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, std::optional<uint64_t> seed = std::nullopt);
