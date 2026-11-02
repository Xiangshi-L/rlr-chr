## Diagnostics of RLR-CHR on SGEMM: residual scores by tertile of Z and the sensitivity to k = c k0.
## Usage: Rscript experiments/run_sgemm_diagnostics.R [--reps=1:12 --n_workers=6 --multipliers=0.5,1,2]

arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(gsub("~\\+~", " ", sub("^--file=", "", arg[1L])))  # Rscript writes spaces as ~+~
source(file.path(here, "..", "rlrchr", "rlr_chr.R"))

flags <- parse_args(c(list(datasets = "sgemm", reps = 1:100, n_workers = 6L),
                      default_config[c("alpha", "n_test", "bins", "levels")],
                      list(multipliers = c(0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00),
                           data = default_data_dir, out = default_out_dir)),
                    lists = c("datasets", "reps", "multipliers"))
cfg <- modifyList(default_config, flags[c("alpha", "n_test", "bins", "levels")])
reps <- flags[["reps"]]
if (sum(abs(flags[["multipliers"]] - 1) < 1e-12) != 1L) stop("--multipliers must contain 1 once")

## Tertiles of z: floor(n/3), floor(2n/3) - floor(n/3) and the rest, in increasing order.
tertile_groups <- function(z) {
  n <- length(z)
  ord <- order(z, na.last = TRUE)
  group <- character(n)
  n_low <- floor(n / 3)
  n_mid <- floor(2 * n / 3) - n_low
  group[ord[seq_len(n_low)]] <- "Low Z"
  group[ord[(n_low + 1L):(n_low + n_mid)]] <- "Mid Z"
  group[ord[(n_low + n_mid + 1L):n]] <- "High Z"
  factor(group, levels = c("Low Z", "Mid Z", "High Z"))
}

## One set of fits per repetition, with R's random numbers in run_repetition's order. The nuisance
## networks are reseeded at every fold, so RLR-CHR at c = 1 equals run_repetition's; the CHR network is
## refitted (the first Keras fit after the split here, the fourth there), so CHR differs. Split-CP,
## Norm-SC, CQR and MCR draw no R random numbers and are not fitted.
diagnostics_rep <- function(rep_i) {
  start_rep <- proc.time()
  alpha <- cfg[["alpha"]]
  d <- split_data(rep_i, ds$X, ds$Y, cfg[["n_test"]])
  n_te <- length(d$Y_te)
  grid <- level_grid(cfg[["levels"]], alpha)
  k_rule <- select_k(d$m, grid$n_levels)

  ## CHR, the reference length
  start_fit <- proc.time()
  q_chr <- fit_quantile_network(d, cfg)
  keras_gc()
  h_y <- chr_histograms(q_chr, bin_edges(d$Y_tr_std, cfg[["bins"]]))
  eps_ca_chr <- runif(d$m)
  eps_te_chr <- runif(n_te)
  chr <- chr_intervals(d, h_y, grid, alpha, eps_ca_chr, eps_te_chr)
  in_chr <- d$Y_te >= chr$lo & d$Y_te <= chr$hi
  len_chr <- chr$hi - chr$lo

  ## RLR-CHR: nuisance fits, calibration scores E and the localizer Z, timed as in run_repetition
  nuis <- fit_nuisances(d, rep_i, cfg)
  fit_seconds <- seconds_since(start_fit)
  edges_r <- bin_edges(nuis$R_chr_tr, cfg[["bins"]])
  start <- proc.time()
  h_r <- chr_histograms(nuis$q, edges_r)
  eps_ca_r <- runif(d$m)
  E_cal <- chr_score_many(nuis$R_chr_ca, h_r$calib, h_r$edges, grid$tau_grid, grid$t0, eps_ca_r)
  time_scores <- seconds_since(start)
  start <- proc.time()
  Z <- localizer_coordinates(nuis$sig_ca, nuis$sig_te)
  time_loc <- seconds_since(start)
  invisible(runif(n_te))  # run_repetition's eps_te_rbc, so that eps_te_rlr is the same draw
  eps_te_rlr <- runif(n_te)

  ## Test residuals R = (Y - mu-hat) / sigma-hat and their CHR scores
  R_te <- (d$Y_te - nuis$mu_te) / nuis$sig_te
  E_te <- chr_score_many((R_te - nuis$r_center) / nuis$r_scale, h_r$test, h_r$edges,
                         grid$tau_grid, grid$t0, eps_te_rlr)
  wsc_setup <- wsc_prepare(d$X_te, seed = 1000 + rep_i)
  groups <- tertile_groups(Z$test)
  gidx <- split(seq_along(groups), groups)

  ## RLR-CHR at each k = c k0, with fits, scores and draws fixed
  mults <- flags[["multipliers"]]
  k_vec <- vapply(mults, function(mult) select_k(d$m, grid$n_levels, mult)$k, integer(1))
  start_sweep <- proc.time()
  k_rows <- vector("list", length(mults))
  for (j in seq_along(mults)) {
    start <- proc.time()
    rlr <- rlr_chr_intervals(E_cal, Z, k_vec[j], h_r, grid, alpha, eps_te_rlr, nuis)
    inference_time <- seconds_since(start) + time_scores + time_loc
    covered <- d$Y_te >= rlr$lo & d$Y_te <= rlr$hi
    len <- rlr$hi - rlr$lo
    cov_g <- vapply(gidx, function(ix) mean(covered[ix]), numeric(1))
    rel_g <- vapply(gidx, function(ix) mean(len[ix]) / mean(len_chr[ix]), numeric(1))
    k_rows[[j]] <- data.frame(
      rep = rep_i, dataset = ds$name, multiplier = mults[j], k = k_vec[j], k0 = k_rule$k0,
      k_min = k_rule$k_min, k_max = k_rule$k_max, T = grid$n_levels, bins = cfg[["bins"]], n_calib = d$m,
      coverage = mean(covered), length = mean(len), wsc = wsc_from_setup(wsc_setup, covered),
      chr_length = mean(len_chr), rel_length = mean(len) / mean(len_chr),
      coverage_lowZ = cov_g[["Low Z"]], coverage_midZ = cov_g[["Mid Z"]], coverage_highZ = cov_g[["High Z"]],
      rel_length_lowZ = rel_g[["Low Z"]], rel_length_midZ = rel_g[["Mid Z"]],
      rel_length_highZ = rel_g[["High Z"]], inference_time = inference_time,
      frac_top_level_rlr = mean(rlr$t >= grid$n_levels), mean_t = mean(as.numeric(rlr$t)))
    if (abs(mults[j] - 1) < 1e-12) c1 <- list(t = as.integer(rlr$t), covered = covered, len = len)
  }
  sweep_seconds <- seconds_since(start_sweep)

  ## Per tertile at c = 1. score_cdf_09 = P(E / T <= 0.9); E = T + 1 lies outside the top set.
  group_rows <- do.call(rbind, lapply(names(gidx), function(gr) {
    ix <- gidx[[gr]]
    qs <- as.numeric(quantile(R_te[ix], c(0.10, 0.50, 0.90), na.rm = TRUE, type = 8))
    data.frame(
      rep = rep_i, dataset = ds$name, group = gr, n = length(ix), k = k_rule$k,
      q10 = qs[1], q50 = qs[2], q90 = qs[3],
      score_cdf_09 = mean(E_te[ix] <= floor(0.9 * grid$n_levels + 1e-9)),
      frac_test_score_gt_T = mean(E_te[ix] > grid$n_levels),
      coverage = mean(c1$covered[ix]), length = mean(c1$len[ix]),
      rel_length = mean(c1$len[ix]) / mean(len_chr[ix]),
      chr_coverage = mean(in_chr[ix]), chr_length = mean(len_chr[ix]))
  }))
  test_rows <- data.frame(
    rep = rep_i, test_row = d$idx_te, group = as.character(groups), Z = Z$test,
    std_residual = R_te, score = as.integer(E_te),
    score_norm = pmin(as.numeric(E_te), grid$n_levels) / grid$n_levels,
    t = c1$t, covered = as.integer(c1$covered), length = c1$len,
    chr_covered = as.integer(in_chr), chr_length = len_chr)
  meta_row <- data.frame(
    rep = rep_i, dataset = ds$name, n_train = nrow(d$X_tr), n_calib = d$m, n_test = n_te,
    T = grid$n_levels, bins = cfg[["bins"]], n_heads = length(quantile_levels), k = k_rule$k,
    chr_t = as.integer(chr$t), fit_seconds = fit_seconds, sweep_seconds = sweep_seconds,
    total_seconds = seconds_since(start_rep))
  keras_gc()
  list(k_rows = do.call(rbind, k_rows), group_rows = group_rows, test_rows = test_rows, meta_row = meta_row)
}

for (key in flags[["datasets"]]) {
  ds <- read_dataset(key, flags[["data"]])
  cat(sprintf("\n%s: N = %d, p = %d; diagnostics, repetitions %d-%d; %s\n", ds$name,
              nrow(ds$X), ncol(ds$X), min(reps), max(reps),
              workers_label(min(flags[["n_workers"]], length(reps)))))
  per_rep <- run_reps(reps, diagnostics_rep, flags[["n_workers"]],
                      export = c("ds", "cfg", "flags", "tertile_groups"))

  stub <- result_path(flags[["out"]], "diagnostics", sprintf("%s_diag_reps%d-%d", key, min(reps), max(reps)))
  part <- function(name) do.call(rbind, lapply(per_rep, `[[`, name))
  write.csv(part("k_rows"), paste0(stub, "_k_long.csv"), row.names = FALSE)
  write.csv(part("group_rows"), paste0(stub, "_scale_group_long.csv"), row.names = FALSE)
  con <- gzfile(paste0(stub, "_scale_test_long.csv.gz"), "w")
  write.csv(part("test_rows"), con, row.names = FALSE)
  close(con)
  write.csv(part("meta_row"), paste0(stub, "_meta.csv"), row.names = FALSE)
  cat(sprintf("saved %s_{k_long, scale_group_long, meta}.csv and _scale_test_long.csv.gz\n", stub))
}
