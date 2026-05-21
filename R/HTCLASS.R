#' HTCLASS
#'
#' Runs the CLASS algorithm on a hadamard transformed dataset.
#'
#' @param X Numeric matrix of predictors (without intercept*).
#' @param y Numeric vector of response variable.
#' @param csv String; path to the data (csv file).
#' @param header Logical; whether the csv files contains a header
#' @param nSample Integer; sample size of the uniform subsample done in CLASS.
#' @param nTimes Integer; number of times/iterations of LASSO in CLASS.
#' @param k Integer; number of rows in the subselection using IBOSS.
#'
#'
#' @useDynLib class, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom stats coef
#' @importFrom stats lm.fit
#' @import glmnet
#' @import data.table
#' @import foreach
#' @import doParallel
#' @import bigmemory
#' @import parallel
#' @import class
#'
#' @return A list with:
#' \itemize{
#' }
#' @export
HTCLASS <- function(X = NULL, y = NULL, csv = NULL, header = FALSE, nSample = -1, nTimes = -1, k = -1) {
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
  N <- nrow(X)

  if (nSample > nrow(X)) {
    stop("nSample cannot be larger than number of rows in X")
  }

  HT_res <- HT(X,y,intercept=FALSE)

  X_HT <- HT_res$X
  y_HT <- HT_res$y

  X_big <- as.big.matrix(x = X_HT, type = "double", backingfile = "X.bin", descriptorfile = "X.desc")
  X_desc <- describe(X_big)

  y_big <- as.big.matrix(x = matrix(y_HT, ncol = 1), type = "double", backingfile = "y.bin", descriptorfile = "y.desc")
  y_desc <- describe(y_big)

  nC <- parallel::detectCores() - 1
  cl <- makeCluster(nC)
  registerDoParallel(cl)

  accumulator <- function(acc, vec) {
    acc + vec
  }
  freq_count <- foreach(i = 1:nTimes, .packages = c("bigmemory", "glmnet", "class"), .combine = accumulator) %dopar% {
    if (i %% 1 == 0) paste(".") # Need an alternative here cuz parallel sessions?

    X_ref <- attach.big.matrix(X_desc)
    y_ref <- attach.big.matrix(y_desc)
    set.seed(42 + i)
    idx <- sample(seq_len(nrow(X_ref)), nSample)
    X_sub <- X_ref[idx, , drop = FALSE]
    y_sub <- y_ref[idx]

    fit <- glmnet::cv.glmnet(x = X_sub, y = y_sub, alpha = 1)
    coefs <- coef(fit, s = "lambda.min")[-1]
    as.numeric(coefs != 0)
  }
  gc() # Should check if this affects runtime
  stopCluster(cl)

  # Should optimise the below
  kboss_res <- kBOSS(X, y, freq_count, k)
  X_final <- kboss_res$X
  y_final <- kboss_res$y
  active_vars <- kboss_res$selected_vars

  intercept_col <- rep(x = 1, times = nrow(X_final))
  X_ols <- cbind(intercept_col, X_final)
  beta_hat <- qr.solve(X_ols, y_final)

  intercept_hat <- beta_hat[1]
  beta_reduced  <- beta_hat[-1]

  final_beta <- rep(0, p)
  final_beta[active_vars + 1] <- beta_reduced

  y_pred <- intercept_hat + X %*% final_beta
  residuals <- y - y_pred

  mse <- mean(residuals^2)
  r_squared <- 1 - sum(residuals^2) / sum((y - mean(y))^2)

  unlink(c("X.bin", "X.desc", "y.bin", "y.desc"))
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
