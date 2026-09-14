# ---------------------------------------------------------------------------
# Tests for SRHT() / SRHT_cpp() (Subsampled Randomized Hadamard Transform).
#
# SRHT is a randomized sketching method: it involves the following steps:
# - Flips the sign of each row with an independent random +/-1,
# - Applies a fast Hadamard transform,
# - Keeps only `k` (out of the zero-padded `n`) rows. 
# Because it has no seed parameter, the *exact* output is never reproducible, so this file tests it the way randomized algorithms have to be tested:
#   - input validation / error handling of the R wrapper (deterministic),
#   - shape/structure invariants that must hold for every draw,
#   - exact algebraic invariants that hold regardless of the randomness,
#   - a statistical invariant (expected squared-norm preservation) checked
#     over many draws with a generous tolerance so it isn't flaky.
# ---------------------------------------------------------------------------

# ---- Input validation (SRHT() wrapper) -------------------------------------

test_that("SRHT requires either {X, y} or csv", {
  expect_error(SRHT(k = 2), "Provide either csv OR")
  expect_error(SRHT(X = matrix(1:4, 2, 2), k = 2), "Provide either csv OR")
  expect_error(SRHT(y = c(1, 2), k = 2), "Provide either csv OR")
})

test_that("SRHT rejects supplying both csv and {X, y}", {
  path <- tempfile(fileext = ".csv")
  on.exit(unlink(path), add = TRUE)
  write.csv(data.frame(x = 1:4, y = 1:4), path, row.names = FALSE)

  expect_error(
    SRHT(X = matrix(1:4, 2, 2), y = c(1, 2), csv = path, k = 2),
    "not both"
  )
})

test_that("SRHT validates the csv path exists", {
  expect_error(
    SRHT(csv = file.path(tempdir(), "does_not_exist.csv"), k = 2),
    "CSV file doesn't exist"
  )
})

test_that("SRHT rejects mismatched row counts and non-numeric inputs", {
  d <- srht_random_fixture(n = 10, p = 2, seed = 11)

  expect_error(SRHT(X = d$X, y = d$y[-1], k = 2), "same number of rows")

  X_char <- matrix(letters[1:20], 10, 2)
  expect_error(SRHT(X = X_char, y = d$y, k = 2), "must be numeric")

  expect_error(SRHT(X = d$X, y = letters[1:10], k = 2), "must be numeric")
})

test_that("SRHT propagates an R-catchable error for a negative k instead of crashing", {
  d <- srht_random_fixture(n = 10, p = 2, seed = 12)
  # No explicit k > 0 check exists in the R wrapper; a negative k is passed
  # straight to the C++ layer, which throws std::length_error while
  # constructing an internal vector. Rcpp's exception translation turns
  # that into a normal, catchable R condition rather than aborting the
  # session -- this test locks in that safety net.
  expect_error(SRHT(X = d$X, y = d$y, k = -3))
})

# ---- Return structure and type coercion -------------------------------------

test_that("SRHT returns a list named X_f, y_f with matching row counts", {
  d <- srht_random_fixture(n = 40, p = 3, seed = 21)
  res <- SRHT(X = d$X, y = d$y, k = 12)

  expect_type(res, "list")
  expect_named(res, c("X_f", "y_f"))
  expect_true(is.matrix(res$X_f))
  expect_equal(nrow(res$X_f), 12)
  expect_equal(length(res$y_f), 12)
  expect_equal(ncol(res$X_f), ncol(d$X))
  expect_true(all(is.finite(res$X_f)))
  expect_true(all(is.finite(res$y_f)))
})

test_that("SRHT coerces a data.frame X via as.matrix", {
  d <- srht_random_fixture(n = 20, p = 2, seed = 22)
  X_df <- as.data.frame(d$X)

  res <- SRHT(X = X_df, y = d$y, k = 5)

  expect_true(is.matrix(res$X_f))
  expect_equal(ncol(res$X_f), ncol(X_df))
  expect_equal(nrow(res$X_f), 5)
})

test_that("SRHT accepts integer-mode X and y and coerces to double", {
  X_int <- matrix(1:20, 10, 2)
  y_int <- as.integer(1:10)

  res <- SRHT(X = X_int, y = y_int, k = 5)

  expect_equal(dim(res$X_f), c(5, 2))
  expect_equal(length(res$y_f), 5)
})

# ---- CSV ingestion -----------------------------------------------------------

test_that("SRHT reads X/y from csv, treating the last column as y", {
  d <- srht_random_fixture(n = 20, p = 3, seed = 31)
  path <- tempfile(fileext = ".csv")
  on.exit(unlink(path), add = TRUE)
  write.csv(cbind(d$X, y = d$y), path, row.names = FALSE)

  res <- SRHT(csv = path, k = 7, header = TRUE)

  expect_equal(dim(res$X_f), c(7, ncol(d$X)))
  expect_equal(length(res$y_f), 7)
})

test_that("SRHT with header = FALSE on a file that has a header row fails as non-numeric", {
  # This documents an actual gotcha in the wrapper: header defaults to
  # FALSE, so fread() reads the header text as a data row, as.matrix()
  # produces a character matrix, and the numeric check then fails.
  d <- srht_random_fixture(n = 10, p = 2, seed = 32)
  path <- tempfile(fileext = ".csv")
  on.exit(unlink(path), add = TRUE)
  write.csv(cbind(d$X, y = d$y), path, row.names = FALSE)

  expect_error(SRHT(csv = path, k = 4), "must be numeric")
})

# ---- Intercept handling -------------------------------------------------------

test_that("SRHT with intercept = TRUE prepends a column of ones when absent", {
  d <- srht_random_fixture(n = 20, p = 2, seed = 41)

  res <- SRHT(X = d$X, y = d$y, k = 5, intercept = TRUE)

  expect_equal(ncol(res$X_f), ncol(d$X) + 1)
})

test_that("SRHT with intercept = TRUE does not duplicate an existing ones column", {
  d <- srht_random_fixture(n = 20, p = 2, seed = 42)
  X_with_ones <- cbind(1, d$X)

  res <- SRHT(X = X_with_ones, y = d$y, k = 5, intercept = TRUE)

  expect_equal(ncol(res$X_f), ncol(X_with_ones))
})

test_that("SRHT's intercept detection tolerance is exactly 1e-9", {
  d <- srht_random_fixture(n = 20, p = 2, seed = 43)

  # Within tolerance: treated as already-ones, not duplicated.
  X_near <- cbind(1 + 1e-10, d$X)
  res_near <- SRHT(X = X_near, y = d$y, k = 5, intercept = TRUE)
  expect_equal(ncol(res_near$X_f), ncol(X_near))

  # Outside tolerance: treated as a real predictor, a new ones column
  # is prepended in front of it.
  X_far <- cbind(1 + 1e-6, d$X)
  res_far <- SRHT(X = X_far, y = d$y, k = 5, intercept = TRUE)
  expect_equal(ncol(res_far$X_f), ncol(X_far) + 1)
})

test_that("SRHT with intercept = FALSE never changes the column count", {
  d <- srht_random_fixture(n = 20, p = 2, seed = 44)
  X_with_ones <- cbind(1, d$X)

  res <- SRHT(X = X_with_ones, y = d$y, k = 5, intercept = FALSE)

  expect_equal(ncol(res$X_f), ncol(X_with_ones))
})

# ---- Shape invariants across problem sizes (exercises Hadamard padding) -----

test_that("SRHT output shape is correct when n is already a power of two", {
  d <- srht_random_fixture(n = 64, p = 5, seed = 51)
  res <- SRHT(X = d$X, y = d$y, k = 20)

  expect_equal(dim(res$X_f), c(20, 5))
  expect_equal(length(res$y_f), 20)
})

test_that("SRHT output shape is correct when n is not a power of two (padded internally)", {
  d <- srht_random_fixture(n = 100, p = 5, seed = 52)
  res <- SRHT(X = d$X, y = d$y, k = 30)

  expect_equal(dim(res$X_f), c(30, 5))
  expect_equal(length(res$y_f), 30)
})

test_that("SRHT with k = 0 returns empty output without error", {
  d <- srht_random_fixture(n = 20, p = 3, seed = 53)
  res <- SRHT(X = d$X, y = d$y, k = 0)

  expect_equal(dim(res$X_f), c(0, 3))
  expect_equal(length(res$y_f), 0)
})

test_that("SRHT with k larger than n still returns k rows (oversampling with replacement)", {
  d <- srht_random_fixture(n = 20, p = 2, seed = 54)
  res <- SRHT(X = d$X, y = d$y, k = 35)

  expect_equal(dim(res$X_f), c(35, 2))
  expect_equal(length(res$y_f), 35)
})

# ---- Algebraic invariants that hold for every draw (not just in expectation) -

test_that("for n = 1, the same random sign hits X and y, so their ratio is preserved exactly", {
  # With a single row there is no Hadamard mixing (bit_ceil(1) == 1), so
  # the only randomness is one +/-1 sign, applied identically to every
  # column of X and to y. That sign cancels out of the ratio regardless
  # of which sign was drawn, making this a deterministic check despite
  # SRHT() having no seed control.
  X <- matrix(c(3.5, -2.0), nrow = 1)
  y <- c(7.0)

  for (i in 1:5) {
    res <- SRHT(X = X, y = y, k = 1)
    expect_equal(res$y_f[1] / res$X_f[1, 1], y[1] / X[1, 1])
    expect_equal(res$X_f[1, 2] / res$X_f[1, 1], X[1, 2] / X[1, 1])
  }
})

test_that("repeated SRHT calls on identical input draw different sketches", {
  # Regression guard against an accidentally fixed/seeded RNG: with no
  # seed control, two calls on the same input should (almost certainly)
  # not produce an identical sketch.
  d <- srht_random_fixture(n = 64, p = 2, seed = 61)

  res1 <- SRHT(X = d$X, y = d$y, k = 20)
  res2 <- SRHT(X = d$X, y = d$y, k = 20)

  expect_false(isTRUE(all.equal(res1$X_f, res2$X_f)))
})

# ---- Statistical correctness: expected squared-norm preservation ------------

test_that("SRHT preserves squared norms in expectation (subspace-embedding property)", {
  # Theory: writing D for the random sign flip and H for the (unnormalized)
  # Hadamard transform, H^T H = n_padded * I and D is orthogonal, so for
  # any column x, ||H D x||^2 = n_padded * ||x||^2. Each of the k sampled
  # entries is drawn uniformly from those n_padded coordinates, so
  # E[sum of the k sampled squared entries] = k * ||x||^2 exactly. This
  # holds for every column of X and for y, and is the key property that
  # makes SRHT a valid sketch for least-squares/regression use downstream.
  #
  # The estimate is noisy for any single call, so this test averages over
  # many independent draws and uses a wide tolerance (calibrated to hold
  # comfortably even though the true relative error typically stays under
  # ~5% for these sizes) to avoid flakiness while still catching a broken
  # transform (e.g. wrong indices, missing sign flip, bad scaling), which
  # would produce errors far larger than the tolerance below.
  set.seed(71)
  n <- 64
  k <- 24
  reps <- 150
  tol <- 0.35

  X <- matrix(rnorm(n * 2), n, 2)
  y <- rnorm(n)

  target_X <- colSums(X^2) * k
  target_y <- sum(y^2) * k

  acc_X <- c(0, 0)
  acc_y <- 0
  for (i in seq_len(reps)) {
    res <- SRHT(X = X, y = y, k = k)
    acc_X <- acc_X + colSums(res$X_f^2)
    acc_y <- acc_y + sum(res$y_f^2)
  }
  mean_X <- acc_X / reps
  mean_y <- acc_y / reps

  expect_equal(mean_X, target_X, tolerance = tol)
  expect_equal(mean_y, target_y, tolerance = tol)
})
