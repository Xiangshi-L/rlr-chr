## RLCP (Hore and Barber, 2025) with the Gaussian kernel, on the engine's splits and WSC setup.
## Usage: Rscript experiments/run_rlcp.R [--datasets=mv,sgemm --reps=1:100 --rlcp_dir=third_party/RLCP]

arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(gsub("~\\+~", " ", sub("^--file=", "", arg[1L])))  # Rscript writes spaces as ~+~
source(file.path(here, "..", "rlrchr", "rlr_chr.R"))

flags <- parse_args(c(list(datasets = names(dataset_registry("main")), reps = 1:100, n_workers = 5L),
                      default_config[c("alpha", "n_test")],
                      list(data = default_data_dir, out = default_out_dir,
                           rlcp_dir = file.path(repo_dir, "third_party", "RLCP"))))
cfg <- modifyList(default_config, flags[c("alpha", "n_test")])
reps <- flags[["reps"]]
rlcp_dir <- normalizePath(flags[["rlcp_dir"]], mustWork = FALSE)

rlcp_commit <- "89dd18becb1d5120534cbdc92da18487a65fbf95"  # the authors' code, pinned in launchers/rlcp.sh
eff_size <- 50L                     # target effective sample size of the authors' bandwidth rule
h_grid <- seq(0.02, 6, by = 0.02)   # bandwidths searched by that rule
n_centers <- 500L                   # Monte Carlo centers of the effective-size estimate

## smoothed_weighted_quantile(v, alpha, w, indices) from utils/methods.R of the authors' code.
load_rlcp_code <- function(dir) {
  file <- file.path(dir, "utils", "methods.R")
  if (!file.exists(file)) stop("RLCP code not found in ", dir, "; run experiments/launchers/rlcp.sh")
  env <- new.env(parent = globalenv())
  sys.source(file, envir = env)
  assign("smoothed_weighted_quantile", env$smoothed_weighted_quantile, envir = globalenv())
}

## Gaussian kernel weights around x_tilde, shifted by the smallest squared distance against underflow.
gaussian_weights <- function(X, x_tilde, h) {
  dx <- sweep(X, 2L, as.numeric(x_tilde), FUN = "-")
  ss <- rowSums(dx * dx)
  exp(-(ss - min(ss)) / (2 * h^2))
}

## Per test point x: x_tilde ~ N(x, h^2 I), weights of the calibration points and x, the weighted quantile.
rlcp_intervals <- function(X_ca, E_ca, X_te, E_te, h, alpha) {
  p <- ncol(X_te)
  X_ca <- as.matrix(X_ca[order(E_ca), , drop = FALSE])
  E <- c(sort(E_ca), Inf)
  first <- c(TRUE, E[-1L] != E[-length(E)])
  E_unique <- E[first]
  positions <- unname(split(seq_along(E), cumsum(first)))  # positions in E of each distinct score
  covered <- numeric(nrow(X_te))
  threshold <- numeric(nrow(X_te))
  for (i in seq_len(nrow(X_te))) {
    x <- X_te[i, ]
    x_tilde <- as.numeric(x) + rnorm(p) * h
    w <- gaussian_weights(rbind(X_ca, x), x_tilde, h)
    q <- smoothed_weighted_quantile(E_unique, alpha, w, positions)
    threshold[i] <- q[1]
    covered[i] <- if (isTRUE(as.logical(q[2]))) E_te[i] <= q[1] else E_te[i] < q[1]
  }
  list(covered = covered, threshold = threshold)
}

## The authors' bandwidth rule: the smallest h of h_grid whose effective size n / ||H||_F^2 - 1
## reaches eff_size; ||H||_F^2 is estimated from n_centers random rows of the kernel matrix H.
rlcp_bandwidth <- function(X, eff_size, h_grid, n_centers) {
  X <- as.matrix(X)
  n <- nrow(X)
  S <- if (n > n_centers) sample.int(n, n_centers) else seq_len(n)
  effective_size <- function(h) {
    r_sum <- 0
    for (i in S) {
      w <- gaussian_weights(X, X[i, ] + rnorm(ncol(X)) * h, h)
      s <- sum(w)
      r_sum <- r_sum + if (is.finite(s) && s > 0) sum((w / s)^2) else 1
    }
    n / (r_sum * (n / length(S))) - 1
  }
  for (h in h_grid) {
    if (effective_size(h) >= eff_size) break
  }
  h
}

rlcp_rep <- function(rep_i) {
  d <- split_data(rep_i, ds$X, ds$Y, cfg[["n_test"]])

  ## f-hat: the point network of Split-CP, at the calibration and test points; scores |Y - f-hat|
  fit <- fit_point_scaled_model(d$X_tr, d$Y_tr, cfg[["epochs"]], cfg[["batch_size"]])
  f <- predict_point_on_splits(fit, list(calib = d$X_ca, test = d$X_te), cfg[["batch_size"]])
  E_ca <- abs(d$Y_ca - f$calib)
  E_te <- abs(d$Y_te - f$test)

  ## The kernel works on covariates standardized with the training means and sds.
  x_mean <- colMeans(d$X_tr)
  x_sd <- apply(d$X_tr, 2L, sd)
  x_sd[!is.finite(x_sd) | x_sd < 1e-12] <- 1
  standardize <- function(X) sweep(sweep(X, 2L, x_mean, `-`), 2L, x_sd, `/`)

  ## Bandwidth from a training subsample of the calibration size m
  n_tune <- min(d$m, nrow(d$X_tr))
  idx_tune <- sample.int(nrow(d$X_tr), n_tune)
  h <- rlcp_bandwidth(standardize(d$X_tr)[idx_tune, , drop = FALSE], eff_size, h_grid,
                      min(n_centers, n_tune))

  start <- proc.time()
  res <- rlcp_intervals(standardize(d$X_ca), E_ca, standardize(d$X_te), E_te, h, cfg[["alpha"]])
  inference_time <- seconds_since(start)
  wsc <- wsc_from_setup(wsc_prepare(d$X_te, seed = 1000 + rep_i), res$covered)

  ## Some thresholds are infinite (unbounded intervals); as the authors do, report the median length.
  data.frame(rep = rep_i, method = "RLCP", dataset = ds$name, coverage = mean(res$covered),
             median_length = 2 * stats::median(res$threshold), wsc = wsc, inference_time = inference_time,
             frac_inf_intervals = mean(res$threshold == Inf))
}

commit <- tryCatch(system2("git", c("-C", shQuote(rlcp_dir), "rev-parse", "HEAD"),
                           stdout = TRUE, stderr = FALSE),
                   warning = function(w) "", error = function(e) "")
if (!identical(commit, rlcp_commit)) {
  stop(rlcp_dir, " is not the RLCP code at commit ", rlcp_commit, "; run experiments/launchers/rlcp.sh")
}
load_rlcp_code(rlcp_dir)

for (key in flags[["datasets"]]) {
  ds <- read_dataset(key, flags[["data"]])
  cat(sprintf("\n%s: N = %d, p = %d; RLCP, repetitions %d-%d, alpha = %.2f; %s\n", ds$name,
              nrow(ds$X), ncol(ds$X), min(reps), max(reps), cfg[["alpha"]],
              workers_label(min(flags[["n_workers"]], length(reps)))))
  per_rep <- run_reps(reps, rlcp_rep, flags[["n_workers"]],
                      export = c("ds", "cfg", "rlcp_dir", "eff_size", "h_grid", "n_centers", "load_rlcp_code",
                                 "gaussian_weights", "rlcp_intervals", "rlcp_bandwidth"),
                      init = function() load_rlcp_code(rlcp_dir))
  out_file <- result_path(flags[["out"]], "rlcp", paste0(key, "_long.csv"))
  write.csv(do.call(rbind, per_rep), out_file, row.names = FALSE)
  cat(sprintf("saved %s\n", out_file))
}
