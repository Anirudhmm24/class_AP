#' Hadamard Transform (HT)
#'
#' This function acts as a wrapper to prepare data and performs the
#' hadamard transform on it. It handles input from either in-memory matrices
#' or CSV files, checks for an intercept, ensures correct data types, and
#' passes the data to the optimized C++ implementation.
#'
#' @param X A numeric matrix of predictors. Required if \code{csv} is NULL.
#'   Must have the same number of rows as \code{y}.
#' @param y A numeric vector of responses. Required if \code{csv} is NULL.
#' @param csv A character string specifying the path to a CSV file. If provided,
#'   \code{X} and \code{y} must be NULL. The function assumes the last column
#'   of the CSV is the response variable \code{y} and all preceding columns
#'   are predictors \code{X}.
#' @param intercept Boolean variable. If \code{TRUE}, the function checks if the first
#'   column of \code{X} is all ones. If not, a column of ones is prepended
#'   to standardizes the intercept term. Defaults to \code{FALSE}.
#' @param header Boolean variable. Indicates if the CSV file contains a header row.
#'   Passed directly to \code{data.table::fread}. Defaults to \code{FALSE}.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{X_f}: Padded and Hadamard-transformed predictor matrix.
#'   \item \code{y_f}: Padded and Hadamard-transformed response vector.
#' }
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(100), nrow = 20, ncol = 5)
#' y <- rnorm(20)
#'
#' res <- HT(X = X, y = y)
#' dim(res$X_f)
#' length(res$y_f)
#'
#' @useDynLib sublime, .registration = TRUE
#' @import Rcpp
#' @importFrom data.table fread
#' @export
HT <- function(X = NULL, y = NULL, csv = NULL, intercept = FALSE, header = FALSE) {

  if (!is.null(csv)) {
    if (!file.exists(csv)) stop("CSV file doesn't exist at given path.")
    if (!is.null(X) || !is.null(y)) stop("Provide either csv OR {X, y}, not both.")

    dat <- data.table::fread(csv, header = header)
    dat <- as.matrix(dat)

    X <- dat[, -ncol(dat), drop = FALSE]
    y <- dat[,  ncol(dat)]
  }
  else {
    if (is.null(X) || is.null(y)) stop("Provide either csv OR {X, y}.")
    if (!is.matrix(X)) X <- as.matrix(X)
  }

  if (nrow(X) != length(y)) stop("X and y must have same number of rows.")
  if (!is.numeric(X) || !is.numeric(y)) stop("X and y must be numeric.")
  if (nrow(X) < 1L) stop("X must have at least one row.")
  if (ncol(X) < 1L) stop("X must have at least one predictor column.")
  if (any(!is.finite(X)) || any(!is.finite(y))) stop("X and y must contain only finite numeric values.")

  if (intercept) {
    first_col_is_ones <- all(abs(X[,1] - 1) < 1e-9)
    if (!first_col_is_ones) {
      X <- cbind(1, X)
    }
  }

  storage.mode(X) <- "double"
  storage.mode(y) <- "double"

  res <- suppressWarnings(HT_cpp(X, y))
  return(res)
}
