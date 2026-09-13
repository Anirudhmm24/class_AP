# sublime

## Overview

sublime is a library which contains implementations of some sub-data selection and randomized sketching algorithms, providing subroutines for speeding up linear regression, lasso and generalized linear models.

These algorithms are:
- IBOSS(): Information-Based Optimal Subdata Selection. Reduces the size of the data matrix by choosing a subset of data, so that the runtime for linear regression is smaller.
- GenIBOSS(): Generalized IBOSS. IBOSS for generalized linear models.
- CLASS(): Combined Lasso And Subdata Selection. Reduces the time taken to run lasso using sub-data selection.
- SRHT(): Randomized sketching method, reduces the runtime for linear regression by preconditioning the data matrix and then sub-sampling.
- HTCLASS(): An variant of CLASS which works by preconditioning the data with a Hadamard matrix in order to improve accuracy for certain datasets.
- CLEAR(): A variant of CLASS which allows for a lesser number of iterations than CLASS, by exiting early according to some criterion.

## Installation

```r
install.packages("sublime")
library(sublime)
```

## Usage

All functions accept either a `X`/`y` pair or a `csv` path (in
which case the last column of the file is treated as the response). Below
are minimal examples for each exported function. See `vignette("sublime")`
for a more detailed walkthrough.

### IBOSS()

Selects a deterministic, information-maximizing subsample for ordinary
linear regression.

```r
library(sublime)

set.seed(42)
X <- matrix(rnorm(20000 * 5), ncol = 5)
beta <- c(2, -1, 0.5, 3, -2)
y <- X %*% beta + rnorm(20000)

res <- IBOSS(X = X, y = y, k = 500)

fit <- lm(res$y_selected ~ res$X_selected)
summary(fit)
```

`IBOSS()` can also read directly from a CSV file:

```r
res <- IBOSS(csv = "data.csv", k = 500, header = TRUE)
```

### GenIBOSS()

Extends IBOSS-style subdata selection to generalized linear models (e.g.
logistic or Poisson regression), using a pilot sample to estimate the
information-weighted design.

```r
set.seed(42)
X <- matrix(rnorm(20000 * 4), ncol = 4)
beta <- c(1, -1, 0.5, 2)
eta <- X %*% beta
p <- 1 / (1 + exp(-eta))
y <- rbinom(nrow(X), 1, p)

fit <- GenIBOSS(
  X = X,
  y = y,
  nSample = 500,
  k = 1000,
  family = binomial(),
  add_logs = TRUE
)

coef(fit$final_model)
```

### CLASS()

Combines Lasso, repeated on sub-sampled data, with IBOSS to speed up variable selection and estimation for high-dimensional linear
regression.

```r
set.seed(42)
n <- 50000
p <- 50
X <- matrix(rnorm(n * p), ncol = p)
beta <- c(rep(2, 5), rep(0, p - 5))
y <- X %*% beta + rnorm(n)

res <- CLASS(
  X = X,
  y = y,
  nSample = 1000,
  nTimes = 50,
  k = 2000
)

res$beta_final
res$mse
res$r_squared
```

### SRHT()

The data matrix is preconditioned with a Hadamard Matrix and a diagonal matrix with 1s and -1s (chosen randomly). Then, rows are randomly subsampled. Performing linear regression on the resultant matrix is much faster.

```r
res <- SRHT(X = X, y = y, k = 2000, intercept = TRUE)
str(res)
```

### HTCLASS()

A variant of `CLASS()` that first applies a Hadamard transform to the
data before running subsampled-Lasso / IBOSS, which can
improve accuracy on certain datasets.

```r
res <- HTCLASS(
  X = X,
  y = y,
  nSample = 1000,
  nTimes = 50,
  k = 2000
)

res$beta_final
res$mse
```

## References
IBOSS: H. Wang, M. Yang, and J. Stufken, “Information-based optimal subdata selection for big
data linear regression,” Journal of the American Statistical Association, vol. 114, no. 525,
pp. 393–405, 2019. https://doi.org/10.1080/01621459.2017.1408468

GenIBOSS: J. Yu, J. Liu, and H. Wang, "Information-based optimal subdata selection for non-linear models," Statistical Papers, vol. 64, no. 4, pp. 1069–1093, Aug. 2023. https://doi.org/10.1007/s00362-023-01430-3

CLASS: R. Singh and J. Stufken, “Subdata selection with a large number of variables,” The New
England Journal of Statistics in Data Science, vol. 1, no. 3, pp. 426–438, 2023. https://doi.org/10.51387/23-NEJSDS36

SRHT: J. A. Tropp, "Improved analysis of the subsampled randomized Hadamard transform," Adv. Adapt. Data Anal., vol. 3, no. 1–2, pp. 115–126, 2011. https://doi.org/10.1142/S1793536911000787
