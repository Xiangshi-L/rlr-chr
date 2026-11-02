## Compiles conformal_core.cpp with Rcpp and binds the R functions of the engine to its exports.

## Builds into cache_dir once; later calls in any process reuse the build. Rcpp rewrites the cache
## index on every call, so processes take turns through a lock directory (stale after 10 minutes).
load_conformal_core <- function(dir, cache_dir = file.path(dir, ".rcpp_cache"),
                                env = parent.frame()) {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  if (file.access(cache_dir, 2L) != 0L) stop("cannot write the build cache ", cache_dir)
  lock <- file.path(cache_dir, "lock")
  while (!dir.create(lock, showWarnings = FALSE)) {
    age <- difftime(Sys.time(), file.mtime(lock), units = "secs")
    if (isTRUE(age > 600)) unlink(lock, recursive = TRUE) else Sys.sleep(0.1)
  }
  on.exit(unlink(lock, recursive = TRUE), add = TRUE)
  Rcpp::sourceCpp(file.path(dir, "conformal_core.cpp"), env = env, cacheDir = cache_dir)
  ## CHR histograms, scores and intervals (one row per point) and the WSC search, used as they are.
  env$quantiles_to_hist_probs_chr <- env$quantiles_to_hist_probs_chr_cpp
  env$chr_score_many <- env$chr_score_many_cpp
  env$chr_interval_many <- env$chr_interval_many_cpp
  env$wsc_find_best_direction <- env$wsc_find_best_direction_cpp
  invisible(NULL)
}

## Level selectors along Z. RBC-CHR's neighbor counts are precomputed in C++ (rbc_chr_precompute);
## the precomputes of RLR-CHR and G2-Residual are R loops in rlr_chr.R (rlr_precompute and
## g2_residual_precompute).
rbc_chr_precompute <- function(E_cal, Z_cal, k) {
  rbc_chr_precompute_cpp(as.integer(E_cal), as.numeric(Z_cal), as.integer(k))
}

rbc_chr_g2_knn_predict_many <- function(E_cal, Z_test, alpha, precomp) {
  rbc_chr_g2_knn_predict_many_cpp(
    as.integer(E_cal), precomp$Z_cal, precomp$kth_dist, precomp$count_neighbors_k,
    precomp$count_neighbors_km1, precomp$k, as.numeric(Z_test), alpha)
}

rlr_chr_select_threshold_many <- function(E_cal, Z_test, alpha, precomp, n_levels) {
  rlr_chr_select_threshold_many_cpp(
    as.integer(E_cal), precomp$Z_cal, precomp$kth_dist, as.integer(precomp$kth_score),
    as.integer(precomp$rank_base), as.integer(precomp$k), as.integer(n_levels),
    as.numeric(Z_test), alpha)
}

g2_residual_knn_predict_one <- function(E_cal, Z_test, alpha, precomp) {
  g2_residual_knn_predict_one_cpp(
    E_cal, precomp$Z_cal, precomp$kth_dist, precomp$sorted_E_k, precomp$sorted_E_km1,
    precomp$k, as.numeric(Z_test), alpha)
}
