#' CLASS
#'
#' Combining LASSO and Subdata Selection
#'
#' Runs the CLASS algorithm on a given dataset. CLASS repeatedly draws
#' uniform subsamples of the data and performs LASSO regression on each
#' subsample. The frequency with which each predictor is selected across
#' the repeated LASSO fits is then used by IBOSS to construct an
#' informative subdata. An ordinary least-squares model is subsequently
#' fitted on the selected subdata, and the resulting coefficients are
#' evaluated on the full dataset.
#'
#' @param X Numeric matrix of predictor variables. Each row corresponds
#'   to an observation and each column to a predictor. The predictor
#'   matrix should not contain an intercept column.
#'   If \code{csv} is provided, \code{X} should be \code{NULL}.
#'
#' @param y Numeric vector containing the response variable. There must
#'   be one response value for each row of \code{X}. If \code{csv} is
#'   provided, \code{y} should be \code{NULL}.
#'
#' @param csv Optional character string specifying the path to a CSV
#'   file containing the dataset. When supplied, the last column of the
#'   CSV file is treated as the response variable and all preceding
#'   columns are treated as predictor variables. \code{X} and \code{y}
#'   must not be supplied when \code{csv} is used.
#'
#' @param header Logical; indicates whether the CSV file specified by
#'   \code{csv} contains a header row.
#'
#' @param nSample Positive integer specifying the number of observations
#'   drawn in each uniform subsample used for the repeated LASSO fits.
#'   It cannot exceed the number of observations in the dataset.
#'
#' @param nTimes Positive integer specifying the total number of repeated
#'   LASSO fits performed by CLASS. The repeated fits are distributed
#'   across the available CPU cores.
#'
#' @param k Positive integer specifying the number of observations to
#'   select in the final subselection performed by IBOSS.
#'
#' @param seed Optional integer specifying the random seed for reproducible
#'   subsampling and cluster execution. Defaults to `NULL`.
#'
#' @return A list containing the following components:
#' \itemize{
#'   \item \code{X_f}: Numeric matrix containing the predictor values
#'   corresponding to the final selected subdata.
#'
#'   \item \code{y_f}: Numeric vector containing the response values
#'   corresponding to the final selected subdata.
#'
#'   \item \code{intercept_hat}: Estimated intercept from the ordinary
#'   least-squares model fitted on the selected subdata.
#'
#'   \item \code{beta_final}: Numeric vector of estimated regression
#'   coefficients for the original predictor variables. Predictors not
#'   selected by the CLASS procedure have coefficient zero.
#'
#'   \item \code{selected_indices}: Indices of the predictor variables
#'   selected by the IBOSS subselection step.
#'
#'   \item \code{feature_counts}: Numeric vector containing the number
#'   of times each predictor was selected across the repeated LASSO fits.
#'
#'   \item \code{mse}: Mean squared error of the final fitted model,
#'   evaluated on the full input dataset.
#'
#'   \item \code{r_squared}: Coefficient of determination (\eqn{R^2})
#'   of the final fitted model, evaluated on the full input dataset.
#' }
#'
#' @details
#' CLASS is designed for large-scale regression problems where repeatedly
#' fitting a model to the full dataset can be computationally expensive.
#' The algorithm uses repeated uniform subsampling and LASSO to identify
#' predictors that are consistently useful across different subsamples.
#'
#' For each of the \code{nTimes} repetitions, CLASS draws \code{nSample}
#' observations uniformly from the full dataset and fits a LASSO model
#' using \code{\link[glmnet]{cv.glmnet}}. The value of the regularization
#' parameter is chosen using the minimum cross-validated error
#' (\code{lambda.min}). A predictor is considered selected in a given
#' repetition when its estimated LASSO coefficient is nonzero.
#'
#' Let \eqn{I_{tj}} denote the indicator that predictor \eqn{j} is
#' selected in repetition \eqn{t}. CLASS computes the selection frequency
#' for each predictor as
#' \deqn{
#' f_j = \sum_{t=1}^{nTimes} I_{tj}.
#' }
#' These frequencies are passed to the IBOSS procedure, which uses them
#' to determine the active predictors and select a final subdata of
#' size \code{k}.
#'
#' An ordinary least-squares model is then fitted using the selected
#' subdata. If the selected predictor matrix is denoted by
#' \eqn{X_f}, the fitted model is
#' \deqn{
#' y_f = \hat{\beta}_0 + X_f\hat{\beta} + \epsilon.
#' }
#' The estimated intercept and coefficients are then used to obtain
#' predictions for all observations in the original dataset.
#'
#' The reported \code{mse} and \code{r_squared} values are calculated
#' using these predictions on the full input dataset rather than only
#' on the selected subdata.
#'
#' CLASS uses parallel processing to distribute the repeated LASSO fits
#' across the available CPU cores. On Unix-alike systems, it uses
#' \code{\link[parallel]{mclapply}} for efficient memory-sharing process
#' forks. On Windows, it falls back to \code{foreach} and
#' \code{bigmemory} to manage parallel execution without excessive memory
#' duplication.
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(500), nrow = 100, ncol = 5)
#' y <- rnorm(100)
#'
#' res <- CLASS(
#'   X = X,
#'   y = y,
#'   nSample = 20,
#'   nTimes = 5,
#'   k = 20
#' )
#'
#' str(res)
#'
#' @references
#' Singh, R. and Stufken, J. (2023).
#' \emph{Subdata Selection With a Large Number of Variables}.
#' The New England Journal of Statistics in Data Science,
#' 1(3), 426--438.
#' doi:10.51387/23-NEJSDS36
#'
#' @author
#' Mathematics Club, IIT Madras
#'
#' @export
#' @useDynLib sublime, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom stats coef
#' @importFrom stats lm.fit
#' @import glmnet
#' @import data.table
#' @import foreach
#' @import doParallel
#' @import bigmemory
#' @import parallel

CLASS <- function(X = NULL, y = NULL, csv = NULL, header = FALSE, nSample = -1, nTimes = -1, k = -1, seed = NULL) {
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) || !is.finite(seed)) {
      stop("seed must be a single finite integer or NULL.")
    }
    seed <- as.integer(seed)
    set.seed(seed)
  }

  if (nSample == -1) {
    stop("Check input CLASS(..., nSample = (pos int), ...")
  }
  if (nTimes == -1) {
    stop("Check input CLASS(..., nTimes = (pos int), ...")
  }
  if (k == -1) {
    stop("Check input CLASS(..., k = (pos int), ...")
  }
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

  p <- ncol(X)

  tss <- sum((y - mean(y))^2)
  if (tss == 0 || all(y == y[1])) {
    intercept_hat <- mean(y)
    final_beta <- rep(0, p)
    return(invisible(list(
      X_f = matrix(numeric(0), nrow = 0, ncol = 0),
      y_f = numeric(0),
      intercept_hat = intercept_hat,
      beta_final = final_beta,
      selected_indices = integer(0),
      feature_counts = rep(0, p),
      mse = 0,
      r_squared = 0
    )))
  }

  if (nSample > nrow(X)) {
    stop("nSample cannot be larger than number of rows in X")
  }

  chk <- Sys.getenv("_R_CHECK_LIMIT_CORES_", "")
  limit_cores <- nzchar(chk) && tolower(chk) != "false"
  if (limit_cores) {
    nC <- 1L
  } else {
    user_cores <- getOption("mc.cores", 1L)
    if (!is.null(user_cores) && is.numeric(user_cores) && user_cores >= 1L) {
      nC <- as.integer(user_cores)
    } else {
      nC <- 1L
    }
  }
  nC <- max(1L, min(nC, as.integer(nTimes)))

  accumulator <- function(acc, vec) {
    acc + vec
  }

  temp <- nTimes %% nC

  if (.Platform$OS.type == "unix") {
    freq_count_list <- parallel::mclapply(X = seq_len(nC), FUN = function(i) {
      local_accumulator <- rep(0, p)
      work <- if (i <= temp) nTimes %/% nC + 1 else nTimes %/% nC

      for (j in seq_len(work)) {
        idx <- sample(seq_len(nrow(X)), nSample)
        X_sub <- X[idx, , drop = FALSE]
        y_sub <- y[idx]

        fit <- glmnet::cv.glmnet(x = X_sub, y = y_sub, alpha = 1)
        coefs <- coef(fit, s = "lambda.min")[-1]
        local_accumulator <- local_accumulator + as.numeric(coefs != 0)
      }
      local_accumulator
    }, mc.cores = nC)
    
    freq_count <- Reduce(accumulator, freq_count_list)

  } else {
    uuid <- paste0(sample(letters, 8, replace = TRUE), collapse = "")
    x_bin <- paste0("X_", uuid, ".bin")
    x_desc <- paste0("X_", uuid, ".desc")
    y_bin <- paste0("y_", uuid, ".bin")
    y_desc <- paste0("y_", uuid, ".desc")
    tmp_path <- tempdir()
    tmp_files <- file.path(tmp_path, c(x_bin, x_desc, y_bin, y_desc))
    on.exit({
      if (exists("cl", inherits = FALSE) && inherits(cl, "cluster")) {
        try(parallel::stopCluster(cl), silent = TRUE)
      }
      if (exists("X_big", inherits = FALSE)) rm(X_big)
      if (exists("y_big", inherits = FALSE)) rm(y_big)
      gc()
      unlink(tmp_files)
    }, add = TRUE)

    X_big <- as.big.matrix(x = X, type = "double", backingfile = x_bin, backingpath = tmp_path, descriptorfile = x_desc)
    X_desc <- describe(X_big)

    y_big <- as.big.matrix(x = matrix(y, ncol = 1), type = "double", backingfile = y_bin, backingpath = tmp_path, descriptorfile = y_desc)
    y_desc <- describe(y_big)

    cl <- makeCluster(nC)
    if (!is.null(seed)) {
      parallel::clusterSetRNGStream(cl, iseed = as.integer(seed))
    }
    parallel::clusterCall(cl, function(lp) .libPaths(lp), .libPaths())
    registerDoParallel(cl)

    freq_count <- foreach(i = seq_len(nC), .packages = c("bigmemory", "glmnet", "sublime"), .combine = accumulator) %dopar% {
      X_ref <- attach.big.matrix(X_desc)
      y_ref <- attach.big.matrix(y_desc)

      local_accumulator <- rep(0, p)
      work <- if (i <= temp) nTimes %/% nC + 1 else nTimes %/% nC

      for (j in seq_len(work)) {
        idx <- sample(seq_len(nrow(X_ref)), nSample)
        X_sub <- X_ref[idx, , drop = FALSE]
        y_sub <- y_ref[idx]

        fit <- glmnet::cv.glmnet(x = X_sub, y = y_sub, alpha = 1)
        coefs <- coef(fit, s = "lambda.min")[-1]
        local_accumulator <- local_accumulator + as.numeric(coefs != 0)
      }
      local_accumulator
    }

    parallel::stopCluster(cl)
    rm(cl)
    rm(X_big, y_big)
    gc()
    unlink(tmp_files)
  }

  kboss_res <- kBOSS(X, y, freq_count, k)
  X_final <- kboss_res$X
  y_final <- kboss_res$y
  active_vars <- kboss_res$selected_vars

  if (length(active_vars) == 0) {
    intercept_hat <- mean(y)
    final_beta <- rep(0, p)
    y_pred <- rep(intercept_hat, nrow(X))
    residuals <- y - y_pred
    mse <- mean(residuals^2)
    r_squared <- 0

    return(invisible(list(
      X_f = X_final,
      y_f = y_final,
      intercept_hat = intercept_hat,
      beta_final = final_beta,
      selected_indices = integer(0),
      feature_counts = freq_count,
      mse = mse,
      r_squared = r_squared
    )))
  }

  intercept_col <- rep(x = 1, times = nrow(X_final))
  X_ols <- cbind(intercept_col, X_final)
  fit_ols <- stats::lm.fit(X_ols, y_final)
  beta_hat <- fit_ols$coefficients
  beta_hat[is.na(beta_hat)] <- 0

  intercept_hat <- beta_hat[1]
  beta_reduced  <- beta_hat[-1]

  final_beta <- rep(0, p)
  final_beta[active_vars + 1] <- beta_reduced

  y_pred <- intercept_hat + X %*% final_beta
  residuals <- y - y_pred

  mse <- mean(residuals^2)
  tss <- sum((y - mean(y))^2)
  rss <- sum(residuals^2)
  r_squared <- if (tss == 0) 0 else 1 - rss / tss

  return(invisible(list(
    X_f = X_final,
    y_f = y_final,
    intercept_hat = intercept_hat,
    beta_final = final_beta,
    selected_indices = active_vars + 1,
    feature_counts = freq_count,
    mse = mse,
    r_squared = r_squared
  )))
}
