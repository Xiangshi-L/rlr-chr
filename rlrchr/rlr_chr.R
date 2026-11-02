## RLR-CHR, RBC-CHR and six comparison methods on the real datasets of the paper.
## Usage: Rscript rlrchr/rlr_chr.R [--datasets=mv,sgemm --reps=1:100 --n_workers=7 --out=results]

## One thread per process (repetitions run in parallel); launchers export these for R's BLAS too.
Sys.setenv(OMP_NUM_THREADS        = "1",
           OPENBLAS_NUM_THREADS   = "1",
           MKL_NUM_THREADS        = "1",
           VECLIB_MAXIMUM_THREADS = "1",
           OMP_DYNAMIC            = "FALSE",
           TF_NUM_INTRAOP_THREADS = "1",
           TF_NUM_INTEROP_THREADS = "1")

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(reticulate)
  library(Rcpp)
})
keras3::config_set_floatx("float32")

## Path of this file, whether it is run with Rscript or source()d.
this_file <- function() {
  for (f in rev(sys.frames())) {
    if (is.character(f$ofile)) {
      path <- f$ofile
      if (!is.null(f$owd) && !grepl("^(/|~|[A-Za-z]:)", path)) path <- file.path(f$owd, path)
      return(normalizePath(path))
    }
  }
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg) > 0L) return(normalizePath(gsub("~\\+~", " ", sub("^--file=", "", arg[1L]))))
  stop("cannot locate rlr_chr.R: run it with Rscript or source() it")
}

engine_file <- this_file()
engine_dir <- dirname(engine_file)
repo_dir <- dirname(engine_dir)
default_data_dir <- file.path(repo_dir, "data", "processed")
default_out_dir <- file.path(repo_dir, "results")
source(file.path(engine_dir, "core_loader.R"), local = TRUE)
load_conformal_core(engine_dir)

## The paper's settings.
default_config <- list(
  alpha = 0.10,
  n_test = 2000L,
  bins = 1000L,     # equal-width histogram bins (bins + 1 edges)
  levels = 1000L,   # number T of nested-set levels (CHR's delta_alpha = 0.001)
  epochs = 200L,
  batch_size = 128L,
  n_folds = 5L      # cross-fitting folds of mu-hat and sigma-hat
)

quantile_levels <- seq(0.01, 0.99, by = 0.01)  # the 99 heads of CHR's real-data experiments

method_names <- c("Split-CP", "Norm-SC", "CQR", "CHR", "MCR", "G2-Residual", "RBC-CHR", "RLR-CHR")

## ====================== Command line ======================
## Flags --name=value; the defaults fix the names and types (NA: required). Flags named in
## `lists` take comma lists, whose integer entries may be ranges A:B. --help prints the flags with
## their defaults. Unknown dataset keys and missing data files are an error.
parse_args <- function(defaults, lists = c("datasets", "reps"),
                       args = commandArgs(trailingOnly = TRUE)) {
  if (any(args %in% c("--help", "-h"))) {
    show <- function(v) {
      if (length(v) == 1L && is.na(v)) return("(required)")
      if (is.integer(v) && length(v) > 1L && all(diff(v) == 1L)) {
        return(sprintf("%d:%d", v[1L], v[length(v)]))
      }
      paste(v, collapse = ",")
    }
    cat(sprintf("--%s=%s\n", names(defaults), vapply(defaults, show, character(1))), sep = "")
    quit(save = "no", status = 0L)
  }
  for (arg in args) {
    kv <- regmatches(arg, regexec("^--([A-Za-z_][A-Za-z0-9_]*)=(.*)$", arg))[[1L]]
    if (length(kv) != 3L) stop("expected --name=value, got: ", arg, call. = FALSE)
    key <- kv[[2L]]
    if (!(key %in% names(defaults))) {
      stop("unknown flag --", key, "; flags: --", paste(names(defaults), collapse = " --"),
           call. = FALSE)
    }
    parts <- if (key %in% lists) strsplit(kv[[3L]], ",", fixed = TRUE)[[1L]] else kv[[3L]]
    value <- parse_value(parts, defaults[[key]])
    if (length(value) == 0L || anyNA(value) || !all(nzchar(value)) ||
        (length(value) > 1L && !(key %in% lists))) {
      stop("bad value for --", key, ": ", kv[[3L]], call. = FALSE)
    }
    defaults[[key]] <- value
  }
  required <- vapply(defaults, function(v) length(v) == 1L && is.na(v), logical(1))
  if (any(required)) {
    stop("missing flag --", paste(names(defaults)[required], collapse = ", --"), call. = FALSE)
  }
  if ("datasets" %in% names(defaults)) dataset_names(defaults[["datasets"]])
  if (all(c("datasets", "data") %in% names(defaults))) {
    files <- file.path(defaults[["data"]], paste0(defaults[["datasets"]], ".csv"))
    missing <- files[!file.exists(files)]
    if (length(missing) > 0L) {
      stop("missing data files: ", paste(missing, collapse = ", "), call. = FALSE)
    }
  }
  defaults
}

parse_value <- function(parts, default) {
  if (!is.numeric(default)) return(parts)
  if (!is.integer(default)) return(suppressWarnings(as.numeric(parts)))
  unlist(lapply(parts, function(p) {
    ab <- suppressWarnings(as.integer(strsplit(p, ":", fixed = TRUE)[[1L]]))
    ok <- grepl("^-?[0-9]+(:-?[0-9]+)?$", p) && (length(ab) == 1L || ab[1L] <= ab[2L])
    if (!ok) NA_integer_ else if (length(ab) == 2L) seq.int(ab[1L], ab[2L]) else ab
  }))
}

## ====================== Data ======================
## Dataset keys and display names of data/<manifest>_manifest.csv ("main" or "additional").
dataset_registry <- function(manifest = "main") {
  reg <- read.csv(file.path(repo_dir, "data", paste0(manifest, "_manifest.csv")),
                  stringsAsFactors = FALSE, check.names = FALSE)
  stats::setNames(as.character(reg[["name"]]), reg[["key"]])
}

## Display names of the dataset keys; an error for a key that is in neither manifest.
dataset_names <- function(keys) {
  registry <- c(dataset_registry("main"), dataset_registry("additional"))
  unknown <- setdiff(keys, names(registry))
  if (length(unknown) > 0L) {
    stop("unknown datasets: ", paste(unknown, collapse = ", "), "; keys: ",
         paste(names(registry), collapse = ", "), call. = FALSE)
  }
  registry[keys]
}

## list(name, X, Y) from <data_dir>/<key>.csv (column `target`; rows with non-finite values dropped).
read_dataset <- function(key, data_dir = default_data_dir) {
  name <- dataset_names(key)[[1L]]
  file <- file.path(data_dir, paste0(key, ".csv"))
  if (!file.exists(file)) stop("missing data file: ", file, call. = FALSE)
  df <- read.csv(file, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  if (!("target" %in% colnames(df))) stop("no column `target` in ", file, call. = FALSE)
  y <- as.numeric(df[["target"]])
  X <- as.matrix(df[, setdiff(colnames(df), "target"), drop = FALSE])
  storage.mode(X) <- "double"
  ok <- is.finite(y) & apply(is.finite(X), 1L, all)
  list(name = name, X = X[ok, , drop = FALSE], Y = y[ok])
}

## Covariates min-max scaled with their training range; a column that is constant on the training
## split is shifted but not scaled, as sklearn's scalers do, so a held-out value of it keeps its units.
scale_covariates <- function(X_tr, X_ca, X_te) {
  xmin <- apply(X_tr, 2L, min)
  xscale <- apply(X_tr, 2L, max) - xmin
  xscale[xscale == 0] <- 1
  rescale <- function(X) sweep(sweep(X, 2, xmin, `-`), 2, xscale, `/`)
  list(train = rescale(X_tr), calib = rescale(X_ca), test = rescale(X_te))
}

## bins + 1 equal-width bin edges over the range of y_tr, padded by 1% of the range on each side.
bin_edges <- function(y_tr, bins) {
  rng <- range(y_tr)
  pad <- 0.01 * diff(rng) + 1e-8
  seq(rng[1] - pad, rng[2] + pad, length.out = bins + 1L)
}

## ====================== Repetitions ======================
## fun(r) for each repetition r of reps, serially or on n_workers PSOCK workers. The workers
## source this file, receive the caller's objects named in export and run init() if given.
run_reps <- function(reps, fun, n_workers, export = character(0), init = NULL) {
  caller <- parent.frame()
  n_workers <- min(as.integer(n_workers), length(reps))
  if (n_workers <= 1L) return(lapply(reps, fun))
  cl <- parallel::makeCluster(n_workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  load_engine <- function(path) {
    source(path)
    NULL
  }
  environment(load_engine) <- globalenv()
  parallel::clusterCall(cl, load_engine, engine_file)
  parallel::clusterExport(cl, export, envir = caller)
  if (!is.null(init)) parallel::clusterCall(cl, init)
  environment(fun) <- globalenv()
  parallel::parLapply(cl, reps, fun)
}

## "1 worker" or "n workers", for the headers the scripts print.
workers_label <- function(n) sprintf("%d worker%s", n, if (n == 1L) "" else "s")

result_path <- function(out, ...) {
  path <- file.path(out, ...)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  path
}

## ====================== Networks ======================
## Frees Keras and Python memory and resets Keras' global state.
keras_gc <- function() {
  keras3::clear_session()
  reticulate::import("gc")$collect()
  invisible(gc())
}

## Seeds TensorFlow and Python, from which Keras draws its initial weights and dropout seeds.
tf_set_seed <- function(s) {
  tensorflow::tf$random$set_seed(as.integer(s))
  reticulate::py_set_seed(s, disable_hash_randomization = FALSE)
}

## Three hidden layers of 64 ReLU units with dropout 0.1, Adam with learning rate 5e-4.
build_dnn_point <- function(input_dim) {
  inputs <- keras_input(shape = input_dim, dtype = "float32")
  outputs <- inputs %>%
    layer_dense(64, activation = "relu") %>% layer_dropout(0.1) %>%
    layer_dense(64, activation = "relu") %>% layer_dropout(0.1) %>%
    layer_dense(64, activation = "relu") %>% layer_dropout(0.1) %>%
    layer_dense(1, activation = "linear")
  model <- keras_model(inputs = inputs, outputs = outputs)
  model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = 5e-4))
  model
}

## One output per quantile level, trained with the pinball loss averaged over the levels.
build_dnn_quantile <- function(input_dim, quantiles) {
  loss_fn <- function(y_true, y_pred) {
    y_true <- op_cast(y_true, dtype = y_pred$dtype)
    q <- op_cast(op_array(as.numeric(quantiles)), dtype = y_pred$dtype)
    ## (B, 1) responses against (1, L) levels: the loss broadcasts over the L heads only.
    y_true <- op_reshape(y_true, c(-1L, 1L))
    q <- op_reshape(q, c(1L, as.integer(length(quantiles))))
    e <- y_true - y_pred
    op_mean(op_maximum(q * e, (q - 1) * e), axis = -1L)
  }
  inputs <- keras_input(shape = input_dim, dtype = "float32")
  outputs <- inputs %>%
    layer_dense(64, activation = "relu") %>% layer_dropout(0.1) %>%
    layer_dense(64, activation = "relu") %>% layer_dropout(0.1) %>%
    layer_dense(64, activation = "relu") %>% layer_dropout(0.1) %>%
    layer_dense(length(quantiles), activation = "linear")
  model <- keras_model(inputs = inputs, outputs = outputs)
  model %>% compile(loss = loss_fn, optimizer = optimizer_adam(learning_rate = 5e-4))
  model
}

## Early stopping on a 10% validation split (patience 10, best weights restored).
fit_with_early_stopping <- function(model, X, y, epochs, batch_size) {
  cb <- list(callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE))
  model %>% fit(X, y, epochs = epochs, batch_size = batch_size,
                validation_split = 0.10, shuffle = TRUE, verbose = 0, callbacks = cb)
}

## Predictions in fixed-size batches (the last one zero-padded): one batch shape for every call.
predict_fixed <- function(model, X, batch_size) {
  tf <- tensorflow::tf
  X <- as.matrix(X)
  n <- as.integer(nrow(X))
  pad <- as.integer((batch_size - (n %% batch_size)) %% batch_size)
  tX <- tf$convert_to_tensor(X, dtype = tf$float32)
  if (pad > 0L) {
    pad_tensor <- tf$zeros(shape = c(pad, as.integer(ncol(X))), dtype = tf$float32)
    tX <- tf$concat(list(tX, pad_tensor), axis = 0L)
  }
  ds <- tf$data$Dataset$from_tensor_slices(tX)$batch(as.integer(batch_size), drop_remainder = TRUE)
  pred <- model %>% predict(ds, verbose = 0)
  pred[seq_len(n), , drop = FALSE]
}

split_rows <- function(x, X_splits) {
  end <- cumsum(vapply(X_splits, nrow, integer(1)))
  start <- c(1L, head(end, -1L) + 1L)
  stats::setNames(lapply(seq_along(end), function(i) x[start[i]:end[i], , drop = FALSE]),
                  names(X_splits))
}

predict_on_splits <- function(model, X_splits, batch_size) {
  split_rows(predict_fixed(model, do.call(rbind, X_splits), batch_size), X_splits)
}

## ====================== Nuisance models ======================
## Center and scale of a standardization: the mean and sd of y (scale 1 if the sd is 0).
standardization <- function(y) {
  ## na.rm = TRUE keeps the numerical path behind the paper's tables: sd(y) can differ in the last bit.
  scale <- sd(y, na.rm = TRUE)
  list(center = mean(y), scale = if (is.finite(scale) && scale > 0) scale else 1)
}

## Point network on the standardized response.
fit_point_scaled_model <- function(X_tr, y_tr, epochs, batch_size) {
  st <- standardization(y_tr)
  model <- build_dnn_point(input_dim = ncol(X_tr))
  fit_with_early_stopping(model, X_tr, (y_tr - st$center) / st$scale, epochs, batch_size)
  list(model = model, center = st$center, scale = st$scale)
}

## Its predictions for each matrix of X_splits, mapped back to the response scale.
predict_point_on_splits <- function(fit, X_splits, batch_size) {
  lapply(predict_on_splits(fit$model, X_splits, batch_size),
         function(p) as.numeric(p) * fit$scale + fit$center)
}

## Cross-fitted mu-hat on n_folds random folds (fold fits seeded with 2024 + fold): fold models and mu_oof.
crossfit_point_ensemble <- function(X_tr, y_tr, n_folds, epochs, batch_size) {
  n <- nrow(X_tr)
  folds <- sample(rep(1:n_folds, length.out = n))
  mu_oof <- rep(NA_real_, n)
  models <- vector("list", n_folds)
  for (fold in seq_len(n_folds)) {
    set.seed(2024 + fold)
    tf_set_seed(2024 + fold)
    idx_in <- which(folds != fold)
    idx_out <- which(folds == fold)
    model <- fit_point_scaled_model(X_tr[idx_in, , drop = FALSE], y_tr[idx_in], epochs, batch_size)
    mu_oof[idx_out] <- predict_point_on_splits(model, list(out = X_tr[idx_out, , drop = FALSE]),
                                               batch_size)$out
    models[[fold]] <- model
  }
  list(models = models, mu_oof = mu_oof, folds = folds)
}

## mu-hat for each matrix of X_splits: the mean of the fold networks' predictions.
predict_ensemble_avg_splits <- function(models, X_splits, batch_size) {
  pred <- lapply(models, predict_point_on_splits, X_splits = X_splits, batch_size = batch_size)
  lapply(stats::setNames(nm = names(X_splits)), function(s) rowMeans(sapply(pred, `[[`, s)))
}

## Out-of-fold predictions (seeds seed_base + fold), and a refit on all points for X_splits_full.
crossfit_predict_oof_and_splits <- function(X_tr, y_tr, X_splits_full, folds, seed_base,
                                            fit_model_fn, predict_splits_fn) {
  oof <- NULL
  for (fold in sort(unique(folds))) {
    idx_in <- which(folds != fold)
    idx_out <- which(folds == fold)
    set.seed(seed_base + fold)
    tf_set_seed(seed_base + fold)
    model <- fit_model_fn(X_tr[idx_in, , drop = FALSE], y_tr[idx_in])
    pred <- as.matrix(predict_splits_fn(model, list(train = X_tr[idx_out, , drop = FALSE]))$train)
    if (is.null(oof)) oof <- matrix(NA_real_, nrow(X_tr), ncol(pred))
    oof[idx_out, ] <- pred
    rm(model, pred)
    keras_gc()
  }

  set.seed(seed_base + max(folds) + 1L)
  tf_set_seed(seed_base + max(folds) + 1L)
  model <- fit_model_fn(X_tr, y_tr)
  full <- predict_splits_fn(model, X_splits_full)
  rm(model)
  keras_gc()
  list(oof = oof, full = full)
}

## Median of |y| (mean of |y|, then 1, if that is not positive).
positive_target_scale <- function(y) {
  y <- as.numeric(y)
  y <- y[is.finite(y)]
  sc <- median(abs(y))
  if (!is.finite(sc) || sc <= 0) sc <- mean(abs(y))
  if (!is.finite(sc) || sc <= 0) sc <- 1
  sc
}

## sigma-hat: median regression of the absolute residuals, on a rescaled target.
fit_sigma_median_abs_scaled <- function(X_tr, abs_resid, epochs, batch_size) {
  target_scale <- positive_target_scale(abs_resid)
  mod <- build_dnn_quantile(input_dim = ncol(X_tr), quantiles = 0.5)
  fit_with_early_stopping(mod, X_tr, as.numeric(abs_resid) / target_scale, epochs, batch_size)
  list(model = mod, target_scale = target_scale)
}

predict_sigma_scaled_on_splits <- function(model_obj, X_splits, batch_size) {
  pred_splits <- predict_on_splits(model_obj$model, X_splits, batch_size)
  lapply(pred_splits, function(x) pmax(as.numeric(x) * model_obj$target_scale, 1e-8))
}

## Floor and cap for sigma-hat from the training absolute residuals.
sigma_bounds <- function(abs_resid) {
  x <- as.numeric(abs_resid)
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(list(floor = 1e-6, cap = Inf))
  med <- median(x)
  q995 <- as.numeric(quantile(x, 0.995, type = 8))
  if (!is.finite(med) || med <= 0) med <- positive_target_scale(x)
  if (!is.finite(q995) || q995 <= 0) q995 <- max(x)
  floor_val <- max(1e-6, med * 1e-3)
  cap_val <- max(1, med * 100, q995 * 20)
  list(floor = floor_val, cap = cap_val)
}

clip_sigma <- function(sig, bounds) {
  sig <- as.numeric(sig)
  sig[!is.finite(sig)] <- bounds$cap
  pmin(pmax(sig, bounds$floor), bounds$cap)
}

## Lowers the cap to max(10 floor, 2.5 median, 2 q99) of the training predictions sig.
lower_sigma_cap <- function(sig, bounds) {
  x <- as.numeric(sig)
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(bounds)
  med <- median(x)
  q99 <- as.numeric(quantile(x, 0.99, type = 8))
  if (!is.finite(med) || med <= 0) med <- positive_target_scale(x)
  if (!is.finite(q99) || q99 <= 0) q99 <- med
  cap <- max(bounds$floor * 10, med * 2.5, q99 * 2)
  if (is.finite(cap) && cap > bounds$floor) bounds$cap <- min(bounds$cap, cap)
  bounds
}

fit_quantile_model <- function(X, y, epochs, batch_size) {
  model <- build_dnn_quantile(input_dim = ncol(X), quantiles = quantile_levels)
  fit_with_early_stopping(model, X, y, epochs, batch_size)
  model
}

## ====================== Steps of one repetition ======================
## Split of repetition r (seeds 1000 + r): n_test test, m = floor(N / 4) calibration, rest training.
split_data <- function(rep_i, X, Y, n_test) {
  set.seed(1000 + rep_i)
  tf_set_seed(1000 + rep_i)
  N <- nrow(X)
  idx_all <- sample.int(N)
  m <- as.integer(floor(0.25 * N))
  if (n_test + m >= N) {
    stop(sprintf("split impossible: N = %d, n_test = %d, m = %d", N, n_test, m))
  }
  idx_te <- idx_all[1:n_test]
  idx_rem <- idx_all[(n_test + 1):N]
  idx_ca <- idx_rem[1:m]
  idx_tr <- idx_rem[(m + 1):length(idx_rem)]

  X_scaled <- scale_covariates(X[idx_tr, , drop = FALSE], X[idx_ca, , drop = FALSE],
                               X[idx_te, , drop = FALSE])
  Y_tr <- Y[idx_tr]
  Y_ca <- Y[idx_ca]
  y_center <- mean(Y_tr)
  y_scale <- sd(Y_tr)
  if (!is.finite(y_scale) || y_scale <= 0) y_scale <- 1
  list(X_tr = X_scaled$train, Y_tr = Y_tr, X_ca = X_scaled$calib, Y_ca = Y_ca,
       X_te = X_scaled$test, Y_te = Y[idx_te], idx_te = idx_te, m = m,
       y_center = y_center, y_scale = y_scale,
       Y_tr_std = (Y_tr - y_center) / y_scale, Y_ca_std = (Y_ca - y_center) / y_scale)
}

## The T levels tau_t = t / T and the index t0 of the level 1 - alpha.
level_grid <- function(n_levels, alpha) {
  n_levels <- as.integer(n_levels)
  list(n_levels = n_levels,
       tau_grid = seq(1 / n_levels, 1, by = 1 / n_levels),
       t0 = max(1L, min(n_levels, ceiling((1 - alpha) * n_levels))))
}

## f-hat of Split-CP, Norm-SC and MCR at the training, calibration and test points.
fit_point_network <- function(d, cfg) {
  fit <- fit_point_scaled_model(d$X_tr, d$Y_tr, cfg[["epochs"]], cfg[["batch_size"]])
  predict_point_on_splits(fit, list(train = d$X_tr, calib = d$X_ca, test = d$X_te),
                          cfg[["batch_size"]])
}

## sigma-hat of Norm-SC: median regression of |Y - f-hat|, clipped to the sigma bounds.
fit_sigma_network <- function(d, f, cfg) {
  abs_resid <- abs(d$Y_tr - f$train)
  bounds <- sigma_bounds(abs_resid)
  fit <- fit_sigma_median_abs_scaled(d$X_tr, abs_resid, cfg[["epochs"]], cfg[["batch_size"]])
  sig <- predict_sigma_scaled_on_splits(fit, list(train = d$X_tr, calib = d$X_ca, test = d$X_te),
                                        cfg[["batch_size"]])
  bounds <- lower_sigma_cap(sig$train, bounds)
  list(calib = clip_sigma(sig$calib, bounds), test = clip_sigma(sig$test, bounds))
}

## A 99-head network on Y_std (CQR and CHR each fit their own): its quantiles at the calibration and
## test points.
fit_quantile_network <- function(d, cfg) {
  model <- fit_quantile_model(d$X_tr, d$Y_tr_std, cfg[["epochs"]], cfg[["batch_size"]])
  predict_on_splits(model, list(calib = d$X_ca, test = d$X_te), cfg[["batch_size"]])
}

## MCR's difficulty at the calibration and test points: crepes' mean of |Y - f-hat| over the 25
## nearest training neighbors.
mcr_difficulty <- function(d, f) {
  crepes_extras <- reticulate::import("crepes.extras")
  np <- reticulate::import("numpy")
  de <- crepes_extras$DifficultyEstimator()
  de$fit(X = np$array(d$X_tr, dtype = "float64"),
         residuals = np$array(abs(d$Y_tr - f$train), dtype = "float64"), k = 25L)
  s_ca <- as.numeric(de$apply(np$array(d$X_ca, dtype = "float64")))
  s_te <- as.numeric(de$apply(np$array(d$X_te, dtype = "float64")))
  list(calib = s_ca, test = s_te)
}

## Nuisance fits of G2-Residual, RBC-CHR and RLR-CHR. mu-hat and sigma-hat (median regression of
## |Y - mu-hat|) are cross-fitted on common folds, which gives their out-of-fold values at the
## training points; at the calibration and test points mu-hat is the mean of the fold networks and
## sigma-hat a refit on all training points. R_std = (Y - mu-hat) / sigma-hat; the 99-head network
## is fitted on all training points to R_chr, the standardized R_std.
fit_nuisances <- function(d, rep_i, cfg) {
  epochs <- cfg[["epochs"]]
  batch_size <- cfg[["batch_size"]]
  X_splits <- list(calib = d$X_ca, test = d$X_te)

  cf <- crossfit_point_ensemble(d$X_tr, d$Y_tr, cfg[["n_folds"]], epochs, batch_size)
  mu <- predict_ensemble_avg_splits(cf$models, X_splits, batch_size)
  cf$models <- NULL
  keras_gc()
  R_oof <- d$Y_tr - cf$mu_oof

  sig_cf <- crossfit_predict_oof_and_splits(
    d$X_tr, abs(R_oof), X_splits, cf$folds, seed_base = 3200L + rep_i * 10L,
    fit_model_fn = function(X, y) fit_sigma_median_abs_scaled(X, y, epochs, batch_size),
    predict_splits_fn = function(model, X_new) predict_sigma_scaled_on_splits(model, X_new, batch_size))
  bounds <- lower_sigma_cap(sig_cf$oof, sigma_bounds(abs(R_oof)))
  sig_tr_raw <- clip_sigma(sig_cf$oof, bounds)
  sig_ca_raw <- clip_sigma(sig_cf$full$calib, bounds)
  sig_te_raw <- clip_sigma(sig_cf$full$test, bounds)

  ## Rescale so that |R| / sigma-hat has median 1 on the training points, then clip again.
  c_star <- median(abs(R_oof) / sig_tr_raw, na.rm = TRUE)
  if (!is.finite(c_star) || c_star <= 0) c_star <- 1
  bounds <- lower_sigma_cap(sig_tr_raw * c_star, bounds)
  sig_tr <- clip_sigma(sig_tr_raw * c_star, bounds)
  sig_ca <- clip_sigma(sig_ca_raw * c_star, bounds)
  sig_te <- clip_sigma(sig_te_raw * c_star, bounds)

  R_std_tr <- R_oof / sig_tr
  R_std_ca <- (d$Y_ca - mu$calib) / sig_ca
  st <- standardization(R_std_tr)
  R_chr_tr <- (R_std_tr - st$center) / st$scale

  seed <- 3300L + rep_i * 10L + cfg[["n_folds"]] + 1L
  set.seed(seed)
  tf_set_seed(seed)
  model <- fit_quantile_model(d$X_tr, R_chr_tr, epochs, batch_size)
  q <- predict_on_splits(model, X_splits, batch_size)
  rm(model)
  keras_gc()
  list(mu_te = mu$test, sig_tr = sig_tr, sig_ca = sig_ca, sig_te = sig_te, R_std_ca = R_std_ca,
       r_center = st$center, r_scale = st$scale, R_chr_tr = R_chr_tr,
       R_chr_ca = (R_std_ca - st$center) / st$scale, q = q)
}

## CHR histograms of the predicted quantiles q$calib and q$test over the bins `edges`.
chr_histograms <- function(q, edges) {
  list(edges = edges, calib = quantiles_to_hist_probs_chr(q$calib, quantile_levels, edges),
       test = quantiles_to_hist_probs_chr(q$test, quantile_levels, edges))
}

## The localizer Z = log sigma-hat. Clipping sigma-hat to its bounds gives Z exact ties; a 1e-9
## uniform jitter (calibration points first) breaks them at random.
localizer_coordinates <- function(sig_cal, sig_test) {
  Z_cal <- log(sig_cal) + runif(length(sig_cal)) * 1e-9
  Z_test <- log(sig_test) + runif(length(sig_test)) * 1e-9
  list(calib = Z_cal, test = Z_test)
}

## ====================== Calibration ======================
## The ceiling((1 - alpha)(m + 1))-th smallest of the m calibration scores and +Inf.
conformal_quantile <- function(E, alpha) {
  sort(c(E, Inf))[ceiling((1 - alpha) * (length(E) + 1L))]
}

## k = round(min(k_max, max(k_min, c k0))), k0 = sqrt(m log T / log m), k_min = 20 (m < 300) or 30,
## k_max = min(300, floor(m / 10)); every kNN-localized method uses c = 1.
select_k <- function(m, n_levels, multiplier = 1) {
  k0 <- sqrt(m * log(n_levels) / log(m))
  k_min <- if (m < 300L) 20L else 30L
  k_max <- min(300L, as.integer(floor(m / 10L)))
  list(k = as.integer(round(min(k_max, max(k_min, multiplier * k0)))), k0 = k0, k_min = k_min, k_max = k_max)
}

## Index of the head at quantile level tau (NA if tau is not one of quantile_levels).
quantile_head <- function(tau) {
  i <- which.min(abs(quantile_levels - tau))
  if (abs(quantile_levels[i] - tau) < 1e-9) i else NA_integer_
}

## Neighborhoods along Z, the same for G2-Residual, RLR-CHR and RBC-CHR (rbc_chr_precompute in C++):
## N_k(i) holds the k calibration points nearest to i, i itself excluded and ties broken by index,
## and the test point enters N_k(i) when it is strictly closer to i than i's k-th neighbor. The two
## R loops below run inside the timed region of their methods.
## G2-Residual: per calibration point i, the distance to its k-th nearest neighbor along Z (ties by
## index) and the sorted scores of its k and k - 1 nearest neighbors.
g2_residual_precompute <- function(E_cal, Z_cal, k) {
  Z_cal <- as.numeric(Z_cal)
  m <- length(E_cal)
  k <- min(k, m - 1L)
  kth_dist <- numeric(m)
  sorted_E_k <- matrix(0, m, k)
  sorted_E_km1 <- matrix(0, m, k - 1L)
  for (i in seq_len(m)) {
    dz <- abs(Z_cal - Z_cal[i])
    dz[i] <- Inf
    nb <- order(dz)[seq_len(k)]
    kth_dist[i] <- dz[nb[k]]
    sorted_E_k[i, ] <- sort(E_cal[nb])
    if (k > 1L) sorted_E_km1[i, ] <- sort(E_cal[nb[-k]])
  }
  list(Z_cal = Z_cal, kth_dist = kth_dist, sorted_E_k = sorted_E_k,
       sorted_E_km1 = sorted_E_km1, k = k)
}

## RLR-CHR: per calibration point i, the distance to and score of its k-th nearest neighbor along Z
## (ties by index) and the local rank A_i = 1 + #{j in N_k(i): E_j <= E_i} without the test point.
rlr_precompute <- function(E_cal, Z_cal, k) {
  Z_cal <- as.numeric(Z_cal)
  m <- length(E_cal)
  k <- min(k, m - 1L)
  kth_dist <- numeric(m)
  kth <- integer(m)
  rank_base <- integer(m)
  for (i in seq_len(m)) {
    dz <- abs(Z_cal - Z_cal[i])
    dz[i] <- Inf
    nb <- order(dz)[seq_len(k)]
    kth_dist[i] <- dz[nb[k]]
    kth[i] <- nb[k]
    rank_base[i] <- 1L + sum(E_cal[nb] <= E_cal[i])
  }
  list(Z_cal = Z_cal, kth_dist = kth_dist, kth_score = E_cal[kth], rank_base = rank_base, k = k)
}

## ====================== Methods ======================
## Each returns the lower and upper ends (lo, hi) of the test intervals.
## Split-CP: f-hat plus or minus the conformal quantile of |Y - f-hat|.
split_cp_intervals <- function(y_ca, f_ca, f_te, alpha) {
  t_hat <- conformal_quantile(abs(y_ca - f_ca), alpha)
  list(lo = f_te - t_hat, hi = f_te + t_hat)
}

## Norm-SC: the same with the scores |Y - f-hat| / sigma-hat.
norm_sc_intervals <- function(y_ca, f_ca, f_te, sig_ca, sig_te, alpha) {
  t_hat <- conformal_quantile(abs(y_ca - f_ca) / sig_ca, alpha)
  list(lo = f_te - t_hat * sig_te, hi = f_te + t_hat * sig_te)
}

## CQR: the alpha/2 and 1 - alpha/2 quantiles q of Y_std, on the response scale, moved by the conformal
## quantile of max(q_lo - Y, Y - q_hi); a negative correction can leave the interval empty.
cqr_intervals <- function(d, q, alpha) {
  head_lo <- quantile_head(alpha / 2)
  head_hi <- quantile_head(1 - alpha / 2)
  if (is.na(head_lo) || is.na(head_hi)) stop("CQR needs quantile heads at alpha/2 and 1 - alpha/2")
  q_ca <- d$y_center + d$y_scale * q$calib
  q_te <- d$y_center + d$y_scale * q$test
  y_ca <- d$Y_ca
  t_hat <- conformal_quantile(pmax(q_ca[, head_lo] - y_ca, y_ca - q_ca[, head_hi]), alpha)
  list(lo = q_te[, head_lo] - t_hat, hi = q_te[, head_hi] + t_hat)
}

## CHR: one level t-hat, the conformal quantile of the calibration scores, for all test points.
chr_intervals <- function(d, h, grid, alpha, eps_ca, eps_te) {
  E <- chr_score_many(d$Y_ca_std, h$calib, h$edges, grid$tau_grid, grid$t0, eps_ca)
  t_hat <- conformal_quantile(E, alpha)
  ## The top nested set is the padded training range, not the real line.
  if (is.finite(t_hat) && t_hat > grid$n_levels) {
    stop("CHR calibration level exceeds T: the top nested set covers only the padded training range")
  }
  t_hat <- as.integer(min(t_hat, grid$n_levels))  # t-hat = +Inf (m < 1 / alpha - 1) becomes T
  ab <- chr_interval_many(h$test, h$edges, grid$tau_grid, grid$t0, eps_te,
                          rep.int(t_hat, length(eps_te)))
  list(lo = d$y_center + d$y_scale * ab[, 1L], hi = d$y_center + d$y_scale * ab[, 2L], t = t_hat)
}

## MCR (crepes 0.9.0): conformal regression of Y - f-hat in 20 equal-size difficulty bins.
mcr_intervals <- function(y_ca, f_ca, f_te, s, alpha) {
  crepes <- reticulate::import("crepes")
  crepes_extras <- reticulate::import("crepes.extras")
  np <- reticulate::import("numpy")
  thresholds <- np$array(
    crepes_extras$binning(np$array(s$calib, dtype = "float64"), bins = 20L)[[2L]],
    dtype = "float64"
  )
  bins_ca <- crepes_extras$binning(np$array(s$calib, dtype = "float64"), bins = thresholds)
  bins_te <- crepes_extras$binning(np$array(s$test, dtype = "float64"), bins = thresholds)
  cr <- crepes$ConformalRegressor()
  cr$fit(residuals = np$array(as.numeric(y_ca - f_ca), dtype = "float64"), bins = bins_ca)
  ends <- cr$predict_int(y_hat = np$array(as.numeric(f_te), dtype = "float64"),
                         bins = bins_te, confidence = 1 - alpha)
  ends <- as.matrix(reticulate::py_to_r(ends))
  list(lo = ends[, 1L], hi = ends[, 2L])
}

## G2-Residual: Guan's G2 rule along Z on the scores E_cal = |R_std|; mu-hat -/+ t sigma-hat.
g2_residual_intervals <- function(E_cal, Z, k, alpha, nuis) {
  precomp <- g2_residual_precompute(E_cal, Z$calib, k)
  t_hat <- vapply(Z$test, function(z) g2_residual_knn_predict_one(E_cal, z, alpha, precomp),
                  numeric(1), USE.NAMES = FALSE)
  list(lo = nuis$mu_te - t_hat * nuis$sig_te, hi = nuis$mu_te + t_hat * nuis$sig_te)
}

## Intervals mu-hat + sigma-hat (r_center + r_scale [a, b]) from CHR intervals [a, b] of R_chr.
residual_intervals <- function(ab, nuis) {
  ab <- nuis$r_center + nuis$r_scale * ab
  list(lo = nuis$mu_te + nuis$sig_te * ab[, 1L], hi = nuis$mu_te + nuis$sig_te * ab[, 2L])
}

## RBC-CHR: the G2 rule along Z on the CHR scores E_cal of R_chr; a level above T (+Inf included)
## becomes T.
rbc_chr_intervals <- function(E_cal, Z, k, h, grid, alpha, eps_te, nuis) {
  precomp <- rbc_chr_precompute(E_cal, Z$calib, k)
  t_rbc <- as.integer(pmin(rbc_chr_g2_knn_predict_many(E_cal, Z$test, alpha, precomp), grid$n_levels))
  ab <- chr_interval_many(h$test, h$edges, grid$tau_grid, grid$t0, eps_te, t_rbc)
  c(residual_intervals(ab, nuis), list(t = t_rbc))
}

## RLR-CHR (Algorithm 1): the largest level e whose local-rank p-value p(e) exceeds alpha.
rlr_chr_intervals <- function(E_cal, Z, k, h, grid, alpha, eps_te, nuis) {
  precomp <- rlr_precompute(E_cal, Z$calib, k)
  t_rlr <- rlr_chr_select_threshold_many(E_cal, Z$test, alpha, precomp, grid$n_levels)
  ab <- chr_interval_many(h$test, h$edges, grid$tau_grid, grid$t0, eps_te, as.integer(t_rlr))
  c(residual_intervals(ab, nuis), list(t = t_rlr))
}

## ====================== Worst-slab coverage (WSC) ======================
## Romano, Sesia and Candes (2020), Section S1.3: B random directions; the worst slab is chosen
## on a random 25% of the test points and evaluated on the other 75%.
wsc_prepare <- function(X_test, seed, B = 1000L, split_frac = 0.25) {
  n <- nrow(X_test)
  if (n < 10L) return(list(valid = FALSE))
  X <- sweep(X_test, 2, colMeans(X_test, na.rm = TRUE), FUN = "-")
  sdv <- apply(X, 2, sd, na.rm = TRUE)
  sdv[!is.finite(sdv) | sdv < 1e-12] <- 1
  X <- sweep(X, 2, sdv, FUN = "/")

  set.seed(seed)
  idx_est <- sample.int(n, size = max(2L, as.integer(floor(split_frac * n))), replace = FALSE)
  idx_eval <- setdiff(seq_len(n), idx_est)
  set.seed(seed + 1000)
  V <- matrix(rnorm(B * ncol(X)), nrow = B, ncol = ncol(X))
  V <- V / sqrt(rowSums(V^2))
  list(valid = TRUE, idx_est = idx_est, idx_eval = idx_eval,
       Z_est = X[idx_est, , drop = FALSE] %*% t(V), Z_eval = X[idx_eval, , drop = FALSE] %*% t(V))
}

## WSC of the coverage indicators `covered` over slabs of mass at least delta.
wsc_from_setup <- function(setup, covered, delta = 0.10) {
  if (!isTRUE(setup$valid)) return(NA_real_)
  covered01 <- as.numeric(as.logical(covered))
  slab <- wsc_find_best_direction(setup$Z_est, covered01[setup$idx_est], delta)
  if (!is.finite(slab$mean_cov) || is.na(slab$dir_idx)) return(NA_real_)
  z_eval <- setup$Z_eval[, slab$dir_idx]
  in_slab <- is.finite(z_eval) & (z_eval >= slab$a) & (z_eval <= slab$b)
  if (any(in_slab)) mean(covered01[setup$idx_eval][in_slab]) else NA_real_
}

wsc_column <- function(delta) {
  if (abs(delta - 0.10) < 1e-12) "wsc" else sprintf("wsc_%03d", as.integer(round(100 * delta)))
}

## Diagnostics: shares of empty CQR intervals, of calibration scores E_cal at level >= T and at
## T + 1 (outside the top nested set), and of test points at level T under RBC-CHR and RLR-CHR;
## median sigma-hat on calibration and test over training; k.
calibration_diagnostic_names <- c(
  "frac_empty_cqr", "frac_cal_score_ge_T", "frac_cal_score_gt_T",
  "frac_top_level_rbc", "frac_top_level_rlr",
  "sig_med_ratio_ca_tr", "sig_med_ratio_te_tr", "k")

calibration_diagnostics <- function(frac_empty_cqr, E_cal, n_levels, t_rbc, t_rlr,
                                    sig_tr, sig_ca, sig_te, k) {
  med_tr <- stats::median(sig_tr)
  out <- c(
    frac_empty_cqr      = as.numeric(frac_empty_cqr),
    frac_cal_score_ge_T = mean(E_cal >= n_levels),
    frac_cal_score_gt_T = mean(E_cal > n_levels),
    frac_top_level_rbc  = mean(t_rbc >= n_levels),
    frac_top_level_rlr  = mean(t_rlr >= n_levels),
    sig_med_ratio_ca_tr = stats::median(sig_ca) / med_tr,
    sig_med_ratio_te_tr = stats::median(sig_te) / med_tr,
    k                   = as.numeric(k))
  out[calibration_diagnostic_names]
}

## Warns when at least 5% of calibration scores are at or above T, or at least half of test
## intervals use level T under RBC-CHR or RLR-CHR (default thresholds).
emit_degenerate_calibration_warning <- function(res, cal_threshold = 0.05, top_threshold = 0.5) {
  row <- res[1L, ]
  flag <- isTRUE(row[["frac_cal_score_ge_T"]] >= cal_threshold) ||
          isTRUE(row[["frac_top_level_rlr"]] >= top_threshold) ||
          isTRUE(row[["frac_top_level_rbc"]] >= top_threshold)
  if (!flag) return(invisible(NULL))
  cat(sprintf(paste0("warning: degenerate calibration (rep %d): calibration scores ",
                     "at level >= T %.3f (at T+1 %.3f); test points at level T: RLR-CHR %.3f, ",
                     "RBC-CHR %.3f; median sigma-hat calib/train %.3g, test/train %.3g\n"),
              as.integer(row[["rep"]]), row[["frac_cal_score_ge_T"]], row[["frac_cal_score_gt_T"]],
              row[["frac_top_level_rlr"]], row[["frac_top_level_rbc"]],
              row[["sig_med_ratio_ca_tr"]], row[["sig_med_ratio_te_tr"]]))
}

seconds_since <- function(start) (proc.time() - start)[["elapsed"]]

## One repetition: per method, coverage, length, WSC at each mass of wsc_deltas (wsc for 0.10,
## wsc_005 for 0.05, ...), inference_time (scoring, precomputation, level selection and test
## intervals; a step shared by several methods counts for each) and the diagnostics.
## keep_points = TRUE attaches attr "points": m, k, coverage indicators and interval ends.
## Seeds of repetition r (set.seed and tf_set_seed unless noted):
##   1000 + r                    split_data; the f-hat, Norm-SC sigma-hat, CQR and CHR networks,
##                               the CHR draws and the folds of fit_nuisances follow in this stream
##   2024 + fold                 the fold fits of mu-hat
##   3200 + 10 r + fold          the fold fits of sigma-hat; 3200 + 10 r + n_folds + 1 its refit
##   3300 + 10 r + n_folds + 1   the residual 99-head network; the later draws follow in this stream
##   1000 + r, 2000 + r          WSC (set.seed only): the split of the test points, the directions
run_repetition <- function(rep_i, X, Y, cfg = default_config, wsc_deltas = 0.10, keep_points = FALSE) {
  alpha <- cfg[["alpha"]]
  if (is.na(quantile_head(alpha / 2)) || is.na(quantile_head(1 - alpha / 2))) {
    stop("--alpha must be a multiple of 0.02 (CQR uses the heads at alpha/2 and 1 - alpha/2); ",
         "run_bins_ablation.R runs other values without CQR", call. = FALSE)
  }
  cat(sprintf("=== repetition %d ===\n", rep_i))
  d <- split_data(rep_i, X, Y, cfg[["n_test"]])
  n_te <- length(d$Y_te)
  grid <- level_grid(cfg[["levels"]], alpha)
  k <- select_k(d$m, grid$n_levels)$k
  iv <- list()
  secs <- stats::setNames(numeric(length(method_names)), method_names)

  f <- fit_point_network(d, cfg)
  start <- proc.time()
  iv[["Split-CP"]] <- split_cp_intervals(d$Y_ca, f$calib, f$test, alpha)
  secs[["Split-CP"]] <- seconds_since(start)

  sig_nsc <- fit_sigma_network(d, f, cfg)
  start <- proc.time()
  iv[["Norm-SC"]] <- norm_sc_intervals(d$Y_ca, f$calib, f$test, sig_nsc$calib, sig_nsc$test, alpha)
  secs[["Norm-SC"]] <- seconds_since(start)

  ## CQR and CHR, each with its own 99-head network
  q_cqr <- fit_quantile_network(d, cfg)
  start <- proc.time()
  iv[["CQR"]] <- cqr_intervals(d, q_cqr, alpha)
  secs[["CQR"]] <- seconds_since(start)

  q_chr <- fit_quantile_network(d, cfg)
  keras_gc()
  edges_y <- bin_edges(d$Y_tr_std, cfg[["bins"]])
  start <- proc.time()
  h_y <- chr_histograms(q_chr, edges_y)
  eps_ca_chr <- runif(d$m)
  eps_te_chr <- runif(n_te)
  iv[["CHR"]] <- chr_intervals(d, h_y, grid, alpha, eps_ca_chr, eps_te_chr)
  secs[["CHR"]] <- seconds_since(start)

  s_mcr <- mcr_difficulty(d, f)
  start <- proc.time()
  iv[["MCR"]] <- mcr_intervals(d$Y_ca, f$calib, f$test, s_mcr, alpha)
  secs[["MCR"]] <- seconds_since(start)

  ## Nuisance fits, CHR scores E_cal of R_chr and the localizer Z
  nuis <- fit_nuisances(d, rep_i, cfg)
  edges_r <- bin_edges(nuis$R_chr_tr, cfg[["bins"]])
  start <- proc.time()
  h_r <- chr_histograms(nuis$q, edges_r)
  eps_ca_r <- runif(d$m)
  E_cal <- chr_score_many(nuis$R_chr_ca, h_r$calib, h_r$edges, grid$tau_grid, grid$t0, eps_ca_r)
  time_scores <- seconds_since(start)
  start <- proc.time()
  Z <- localizer_coordinates(nuis$sig_ca, nuis$sig_te)
  time_loc <- seconds_since(start)

  start <- proc.time()
  iv[["G2-Residual"]] <- g2_residual_intervals(abs(nuis$R_std_ca), Z, k, alpha, nuis)
  secs[["G2-Residual"]] <- seconds_since(start) + time_loc

  start <- proc.time()
  eps_te_rbc <- runif(n_te)
  iv[["RBC-CHR"]] <- rbc_chr_intervals(E_cal, Z, k, h_r, grid, alpha, eps_te_rbc, nuis)
  secs[["RBC-CHR"]] <- seconds_since(start) + time_scores + time_loc

  start <- proc.time()
  eps_te_rlr <- runif(n_te)
  iv[["RLR-CHR"]] <- rlr_chr_intervals(E_cal, Z, k, h_r, grid, alpha, eps_te_rlr, nuis)
  secs[["RLR-CHR"]] <- seconds_since(start) + time_scores + time_loc

  diagnostics <- calibration_diagnostics(
    frac_empty_cqr = mean(iv[["CQR"]]$hi < iv[["CQR"]]$lo), E_cal = E_cal,
    n_levels = grid$n_levels, t_rbc = iv[["RBC-CHR"]]$t,
    t_rlr = iv[["RLR-CHR"]]$t, sig_tr = nuis$sig_tr, sig_ca = nuis$sig_ca,
    sig_te = nuis$sig_te, k = k)

  ## Coverage, length (an empty CQR interval has length 0) and WSC
  lo <- sapply(method_names, function(mt) iv[[mt]]$lo)
  hi <- sapply(method_names, function(mt) iv[[mt]]$hi)
  covered <- d$Y_te >= lo & d$Y_te <= hi
  width <- hi - lo
  width[, "CQR"] <- pmax(0, width[, "CQR"])
  per_method <- function(fun) vapply(method_names, fun, numeric(1), USE.NAMES = FALSE)
  res <- data.frame(rep = rep_i, method = method_names,
                    coverage = per_method(function(mt) mean(covered[, mt])),
                    length = per_method(function(mt) mean(width[, mt])))
  wsc_setup <- wsc_prepare(d$X_te, seed = 1000 + rep_i)
  for (delta in wsc_deltas) {
    res[[wsc_column(delta)]] <- per_method(function(mt) wsc_from_setup(wsc_setup, covered[, mt], delta))
  }
  res[["inference_time"]] <- unname(secs)
  for (nm in calibration_diagnostic_names) res[[nm]] <- diagnostics[[nm]]
  emit_degenerate_calibration_warning(res)
  keras_gc()

  if (keep_points) attr(res, "points") <- list(m = d$m, k = k, covered = covered, lo = lo, hi = hi)
  res
}

## ====================== Main ======================
## run_repetition over reps on one dataset: all rows, with the dataset's name after rep and method.
run_dataset <- function(key, reps, n_workers, cfg = default_config, wsc_deltas = 0.10,
                        data_dir = default_data_dir) {
  ds <- read_dataset(key, data_dir)
  n_workers <- min(n_workers, length(reps))
  cat(sprintf(paste0("\n%s: N = %d, p = %d; repetitions %d-%d, alpha = %.2f, n_test = %d, ",
                     "%d bins, T = %d, k = %d; %s\n"),
              ds$name, nrow(ds$X), ncol(ds$X), min(reps), max(reps), cfg[["alpha"]],
              cfg[["n_test"]], cfg[["bins"]], cfg[["levels"]],
              select_k(floor(0.25 * nrow(ds$X)), cfg[["levels"]])$k, workers_label(n_workers)))
  rows <- run_reps(reps, function(rep_i) run_repetition(rep_i, ds$X, ds$Y, cfg, wsc_deltas),
                   n_workers, export = c("ds", "cfg", "wsc_deltas"))
  ## Worker output is discarded, so their warnings are repeated here.
  if (n_workers > 1L) for (res in rows) emit_degenerate_calibration_warning(res)
  long <- do.call(rbind, rows)
  cbind(long[c("rep", "method")], dataset = ds$name, long[setdiff(names(long), c("rep", "method"))])
}

## Writes <out>/main/<key>_long.csv for each dataset (default: every key of data/main_manifest.csv).
main <- function() {
  flags <- parse_args(c(list(datasets = names(dataset_registry("main")), reps = 1:100, n_workers = 7L),
                        default_config[c("alpha", "n_test", "bins", "levels")],
                        list(data = default_data_dir, out = default_out_dir)))
  cfg <- modifyList(default_config, flags[c("alpha", "n_test", "bins", "levels")])
  for (key in flags[["datasets"]]) {
    long <- run_dataset(key, flags[["reps"]], flags[["n_workers"]], cfg, data_dir = flags[["data"]])
    avg <- aggregate(long[c("coverage", "length", "wsc", "inference_time")],
                     by = long["method"], FUN = mean, na.rm = TRUE)
    print(avg[match(method_names, avg[["method"]]), ], row.names = FALSE, digits = 4)
    out_file <- result_path(flags[["out"]], "main", paste0(key, "_long.csv"))
    write.csv(long, out_file, row.names = FALSE)
    cat(sprintf("saved %s\n", out_file))
  }
}

if (sys.nframe() == 0L) main()
