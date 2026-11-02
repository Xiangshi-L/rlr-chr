## Coverage of the eight methods in strata of test points by the distance to the k-th nearest training point.
## Usage: Rscript experiments/run_density.R [--datasets=mv,sgemm --reps=1:100 --k=20]

arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(gsub("~\\+~", " ", sub("^--file=", "", arg[1L])))  # Rscript writes spaces as ~+~
source(file.path(here, "..", "rlrchr", "rlr_chr.R"))

flags <- parse_args(c(list(datasets = names(dataset_registry("main")), reps = 1:100, n_workers = 6L),
                      default_config[c("alpha", "n_test", "bins", "levels")],
                      list(k = 20L, n_strata = 5L, data = default_data_dir, out = default_out_dir)))
cfg <- modifyList(default_config, flags[c("alpha", "n_test", "bins", "levels")])
reps <- flags[["reps"]]

## r_k(x), the exact distance to the k-th nearest training point in the training-standardized covariates
## (constant columns dropped).
knn_radius <- function(X_tr, X_te, k) {
  mu <- colMeans(X_tr)
  sdv <- apply(X_tr, 2, stats::sd)
  keep <- is.finite(sdv) & sdv > 1e-12
  S_tr <- sweep(sweep(X_tr[, keep, drop = FALSE], 2, mu[keep], "-"), 2, sdv[keep], "/")
  S_te <- sweep(sweep(X_te[, keep, drop = FALSE], 2, mu[keep], "-"), 2, sdv[keep], "/")
  tS <- t(S_tr)
  r <- numeric(nrow(S_te))
  for (j in seq_len(nrow(S_te))) {
    d2 <- colSums((tS - S_te[j, ])^2)
    r[j] <- sqrt(sort(d2, partial = k)[k])
  }
  list(r = r, p_used = sum(keep), dropped = which(!keep))
}

## n_strata equal strata by decreasing proxy, ties by test position; stratum 1 is the sparsest.
density_strata <- function(proxy, n_strata) {
  rk <- integer(length(proxy))
  rk[order(-proxy, seq_along(proxy))] <- seq_along(proxy)
  as.integer(ceiling(n_strata * rk / length(proxy)))
}

density_rep <- function(rep_i) {
  res <- run_repetition(rep_i, ds$X, ds$Y, cfg, keep_points = TRUE)
  pts <- attr(res, "points")
  d <- split_data(rep_i, ds$X, ds$Y, cfg[["n_test"]])  # the same split, for the covariates
  r_k <- knn_radius(d$X_tr, d$X_te, flags[["k"]])
  proxy <- r_k$r
  stratum <- density_strata(proxy, flags[["n_strata"]])
  width <- pmax(pts$hi - pts$lo, 0)  # an empty CQR interval has length 0, as in the engine

  groups <- c(as.character(seq_len(flags[["n_strata"]])), "all")
  strata <- do.call(rbind, lapply(method_names, function(mt) {
    do.call(rbind, lapply(groups, function(s) {
      w <- if (s == "all") rep(TRUE, length(stratum)) else stratum == as.integer(s)
      data.frame(rep = rep_i, method = mt, dataset = ds$name, proxy = paste0("r", flags[["k"]]),
                 stratum = s, n = sum(w), coverage = mean(pts$covered[w, mt]), length = mean(width[w, mt]),
                 mean_proxy = mean(proxy[w]))
    }))
  }))
  engine_long <- cbind(res[c("rep", "method")], dataset = ds$name,
                       res[c("coverage", "length", "wsc", "inference_time")])
  list(strata = strata, engine_long = engine_long, n_train = nrow(d$X_tr), m = pts$m, k = pts$k,
       p_used = r_k$p_used, dropped = r_k$dropped)
}

for (key in flags[["datasets"]]) {
  ds <- read_dataset(key, flags[["data"]])
  cat(sprintf("\n%s: N = %d, p = %d; density strata, repetitions %d-%d; %s\n", ds$name,
              nrow(ds$X), ncol(ds$X), min(reps), max(reps),
              workers_label(min(flags[["n_workers"]], length(reps)))))
  per_rep <- run_reps(reps, density_rep, flags[["n_workers"]],
                      export = c("ds", "cfg", "flags", "knn_radius", "density_strata"))

  stub <- result_path(flags[["out"]], "density", key)
  part <- function(name) do.call(rbind, lapply(per_rep, `[[`, name))
  write.csv(part("strata"), paste0(stub, "_strata.csv"), row.names = FALSE)
  write.csv(part("engine_long"), paste0(stub, "_engine_long.csv"), row.names = FALSE)
  ## One row: the split sizes and the localized methods' k (first repetition), the proxy and the settings.
  first <- per_rep[[1L]]
  dropped <- colnames(ds$X)[first$dropped]
  meta <- data.frame(key = key, dataset = ds$name, N = nrow(ds$X), p = ncol(ds$X), rep_first = min(reps),
                     rep_last = max(reps), n_train = first$n_train, n_calib = first$m, k = first$k,
                     p_used_by_proxy = first$p_used,
                     proxy_dropped_columns = if (length(dropped) > 0L) paste(dropped, collapse = ";") else NA,
                     n_test = cfg[["n_test"]], alpha = cfg[["alpha"]], bins = cfg[["bins"]],
                     levels = cfg[["levels"]], proxy_k = flags[["k"]], n_strata = flags[["n_strata"]])
  write.csv(meta, paste0(stub, "_meta.csv"), row.names = FALSE)
  cat(sprintf("saved %s_{strata, engine_long, meta}.csv\n", stub))
}
