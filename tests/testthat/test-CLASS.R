test_that("CLASS produces expected output structure", {
  set.seed(42)
  N <- 100
  p <- 10
  
  X <- matrix(rnorm(N * p), N, p)
  y <- X[, 1] * 2 + X[, 2] * -2 + rnorm(N)
  
  res <- CLASS(X, y, nSample = 50, nTimes = 5, k = 20)
  
  expect_type(res, "list")
  expect_true("X_f" %in% names(res))
  expect_true("y_f" %in% names(res))
  expect_true("feature_counts" %in% names(res))
  expect_equal(length(res$feature_counts), p)
  expect_equal(nrow(res$X_f), length(res$y_f))
  expect_lte(nrow(res$X_f), 20)
  expect_gte(nrow(res$X_f), 0)
  expect_gt(nrow(res$X_f), 0)
})

test_that("CLASS runs without errors on mock OS", {
  set.seed(42)
  N <- 100
  p <- 10
  X <- matrix(rnorm(N * p), N, p)
  y <- X[, 1] * 2 + X[, 2] * -2 + rnorm(N)
  
  # Basic execution run to cover the active OS code path
  expect_error(CLASS(X, y, nSample = 50, nTimes = 2, k = 20), NA)
})

test_that("CLASS validates required arguments", {
  N <- 50; p <- 5
  X <- matrix(rnorm(N * p), N, p)
  y <- rnorm(N)
  
  expect_error(CLASS(X, y, nTimes = 5, k = 10), "nSample")
  expect_error(CLASS(X, y, nSample = 20, k = 10), "nTimes")
  expect_error(CLASS(X, y, nSample = 20, nTimes = 5), "k")
})

test_that("CLASS validates dimensions and types", {
  N <- 50; p <- 5
  X <- matrix(rnorm(N * p), N, p)
  y <- rnorm(N)
  
  expect_error(CLASS(X, y[1:10], nSample = 20, nTimes = 2, k = 10), "X and y must have same number of rows")
  expect_error(CLASS(X, as.character(y), nSample = 20, nTimes = 2, k = 10), "X and y must be numeric")
  expect_error(CLASS(X, y, nSample = 100, nTimes = 2, k = 10), "nSample cannot be larger than number of rows in X")
})

test_that("CLASS validates CSV input conflicts", {
  N <- 50; p <- 5
  X <- matrix(rnorm(N * p), N, p)
  y <- rnorm(N)
  
  expect_error(CLASS(csv = "nonexistent_file.csv", nSample = 20, nTimes = 2, k = 10), "CSV file doesn't exist at given path")
  
  tmp_csv <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_csv), add = TRUE)
  write.table(cbind(X, y), tmp_csv, row.names = FALSE, col.names = FALSE, sep = ",")
  
  expect_error(CLASS(X = X, csv = tmp_csv, nSample = 20, nTimes = 2, k = 10), "Provide either csv OR \\{X, y\\}, not both")
  expect_error(CLASS(nSample = 20, nTimes = 2, k = 10), "Provide either csv OR \\{X, y\\}")
})

test_that("CLASS works with CSV input", {
  set.seed(42)
  N <- 50; p <- 5
  X <- matrix(rnorm(N * p), N, p)
  y <- X[, 1] * 2 + X[, 2] * -2 + rnorm(N)
  dat <- cbind(X, y)
  
  tmp_csv <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp_csv), add = TRUE)
  write.table(dat, tmp_csv, row.names = FALSE, col.names = FALSE, sep = ",")
  
  res <- CLASS(csv = tmp_csv, header = FALSE, nSample = 20, nTimes = 2, k = 10)
  
  expect_type(res, "list")
  expect_true("X_f" %in% names(res))
  expect_true("y_f" %in% names(res))
  expect_equal(length(res$feature_counts), p)
  expect_equal(nrow(res$X_f), length(res$y_f))
  expect_lte(nrow(res$X_f), 10)
  expect_gte(nrow(res$X_f), 0)
  expect_gt(nrow(res$X_f), 0)
})
