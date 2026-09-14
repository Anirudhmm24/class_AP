#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

#include "KMeans.hpp"

std::vector<int> KMeans2(const std::vector<double>& counts) {
    int q = counts.size();
    if (q == 0) return {};

    // Basic two-means algorithm:
    // 1. Initialize the two centers with the maximum and minimum values.
    // 2. Assign every value to its nearest center.
    // 3. Recalculate each center as the mean of its assigned values.
    // 4. Keep an unchanged center if its cluster is empty.
    // 5. Repeat until both centers stabilize or the iteration limit is reached.
    const auto minmax = std::minmax_element(counts.begin(), counts.end());
    double m1 = *minmax.second;
    double m2 = *minmax.first;

    constexpr double tolerance = 1e-3;
    constexpr int max_iterations = 50;

    for (int iter = 0; iter < max_iterations; ++iter) {
        double m1s = 0, m2s = 0;
        int m1c = 0, m2c = 0;
        for (int i = 0; i < q; i++) {
            if (std::abs(m1 - counts[i]) < std::abs(m2 - counts[i])) {
                m1s += counts[i];
                m1c += 1;
            }
            else {
                m2s += counts[i];
                m2c += 1;
            }
        }
        // An empty cluster has no mean, so retain its previous center.
        const double new_m1 = m1c == 0 ? m1 : m1s / m1c;
        const double new_m2 = m2c == 0 ? m2 : m2s / m2c;
        const bool converged = std::abs(new_m1 - m1) <= tolerance &&
            std::abs(new_m2 - m2) <= tolerance;
        m1 = new_m1;
        m2 = new_m2;
        if (converged) break;
    }

    // Return the indices assigned to the higher-valued cluster.
    std::vector<int> active_vars;
    if (m1 < m2) {
        std::swap(m1, m2);
    }
    for (int i = 0; i < q; i++) {
        if (std::abs(m1 - counts[i]) < std::abs(m2 - counts[i])) active_vars.push_back(i);
    }

    return active_vars;
}

void KMeans(const std::vector<double>& counts, std::vector<int>& active_vars) {
    int q = counts.size();
    if (q == 0) return;

    // This is the in-place-output version of KMeans2; the clustering steps
    // and convergence rules are the same, but results are appended to the
    // caller-provided active_vars vector.
    const auto minmax = std::minmax_element(counts.begin(), counts.end());
    double m1 = *minmax.second;
    double m2 = *minmax.first;

    constexpr double tolerance = 1e-3;
    constexpr int max_iterations = 50;

    for (int iter = 0; iter < max_iterations; ++iter) {
        double m1s = 0, m2s = 0;
        int m1c = 0, m2c = 0;
        for (int i = 0; i < q; i++) {
            if (std::abs(m1 - counts[i]) < std::abs(m2 - counts[i])) {
                m1s += counts[i];
                m1c += 1;
            }
            else {
                m2s += counts[i];
                m2c += 1;
            }
        }
        const double new_m1 = m1c == 0 ? m1 : m1s / m1c;
        const double new_m2 = m2c == 0 ? m2 : m2s / m2c;
        const bool converged = std::abs(new_m1 - m1) <= tolerance &&
            std::abs(new_m2 - m2) <= tolerance;
        m1 = new_m1;
        m2 = new_m2;
        if (converged) break;
    }

    if (m1 < m2) {
        std::swap(m1, m2);
    }
    for (int i = 0; i < q; i++) {
        if (std::abs(m1 - counts[i]) < std::abs(m2 - counts[i])) active_vars.push_back(i);
    }
}