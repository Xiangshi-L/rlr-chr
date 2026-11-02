## CHR and RLR-CHR at several numbers of histogram bins, with CQR for reference, from one set of fits.
## Usage: Rscript experiments/run_bins_ablation.R --datasets=blog [--bins=100,1000 --alpha=0.10]

arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(gsub("~\\+~", " ", sub("^--file=", "", arg[1L])))  # Rscript writes spaces as ~+~
source(file.path(here, "..", "rlrchr", "rlr_chr.R"))

flags <- parse_args(c(list(datasets = NA_character_, reps = 1:100, n_workers = 4L),
                      default_config[c("alpha", "n_test")],
                      list(bins = c(100L, 1000L), levels = 1000L, data = default_data_dir,
                           out = default_out_dir)),
                    lists = c("datasets", "reps", "bins"))
cfg <- modifyList(default_config, flags[c("alpha", "n_test", "levels")])
reps <- flags[["reps"]]
bins_list <- sort(unique(flags[["bins"]]))

## ablations/bins_<key>_long.csv at the paper's alpha = 0.10, else bins_alpha<100 alpha>_<key>_long.csv
out_name <- function(key) {
  if (cfg[["alpha"]] == 0.10) return(file.path("ablations", paste0("bins_", key, "_long.csv")))
  file.path("ablations", sprintf("bins_alpha%03d_%s_long.csv", as.integer(round(100 * cfg[["alpha"]])), key))
}

## Networks, Z and draws are fixed; only the bin edges change with the number of bins.
bins_rep <- function(rep_i) {
  alpha <- cfg[["alpha"]]
  d <- split_data(rep_i, ds$X, ds$Y, cfg[["n_test"]])
  grid <- level_grid(cfg[["levels"]], alpha)
  k <- select_k(d$m, grid$n_levels)$k

  ## CQR needs alpha/2 and 1 - alpha/2 among the quantile levels of the 99 heads.
  with_cqr <- !is.na(quantile_head(alpha / 2)) && !is.na(quantile_head(1 - alpha / 2))
  if (with_cqr) {
    q_cqr <- fit_quantile_network(d, cfg)
    keras_gc()
    cqr <- cqr_intervals(d, q_cqr, alpha)
  }
  q_chr <- fit_quantile_network(d, cfg)
  keras_gc()
  nuis <- fit_nuisances(d, rep_i, cfg)
  Z <- localizer_coordinates(nuis$sig_ca, nuis$sig_te)
  eps_ca_chr <- runif(d$m)
  eps_te_chr <- runif(length(d$Y_te))
  eps_ca_r <- runif(d$m)
  eps_te_r <- runif(length(d$Y_te))
  wsc_setup <- wsc_prepare(d$X_te, seed = 1000 + rep_i)

  summary_row <- function(method, bins, iv, width = iv$hi - iv$lo, frac_empty_cqr = NA_real_) {
    covered <- d$Y_te >= iv$lo & d$Y_te <= iv$hi
    row <- data.frame(rep = rep_i, method = method, bins = bins, coverage = mean(covered),
                      length = mean(width), frac_empty_cqr = frac_empty_cqr)
    for (delta in c(0.05, 0.10, 0.20)) row[[wsc_column(delta)]] <- wsc_from_setup(wsc_setup, covered, delta)
    row
  }
  rows <- list()
  ## An empty CQR interval has length 0; frac_empty_cqr is the share of empty CQR intervals.
  if (with_cqr) {
    rows[[1L]] <- summary_row("CQR", NA_integer_, cqr, pmax(cqr$hi - cqr$lo, 0), mean(cqr$hi < cqr$lo))
  }
  for (bins in bins_list) {
    h_y <- chr_histograms(q_chr, bin_edges(d$Y_tr_std, bins))
    chr <- chr_intervals(d, h_y, grid, alpha, eps_ca_chr, eps_te_chr)
    h_r <- chr_histograms(nuis$q, bin_edges(nuis$R_chr_tr, bins))
    E_cal <- chr_score_many(nuis$R_chr_ca, h_r$calib, h_r$edges, grid$tau_grid, grid$t0, eps_ca_r)
    rlr <- rlr_chr_intervals(E_cal, Z, k, h_r, grid, alpha, eps_te_r, nuis)
    rows <- c(rows, list(summary_row("CHR", bins, chr), summary_row("RLR-CHR", bins, rlr)))
  }
  keras_gc()
  do.call(rbind, rows)
}

for (key in flags[["datasets"]]) {
  ds <- read_dataset(key, flags[["data"]])
  cat(sprintf("\n%s: N = %d, p = %d; bins %s, T = %d, alpha = %.2f, repetitions %d-%d; %s\n",
              ds$name, nrow(ds$X), ncol(ds$X), paste(bins_list, collapse = ","), cfg[["levels"]],
              cfg[["alpha"]], min(reps), max(reps), workers_label(min(flags[["n_workers"]], length(reps)))))
  per_rep <- run_reps(reps, bins_rep, flags[["n_workers"]], export = c("ds", "cfg", "bins_list"))
  out_file <- result_path(flags[["out"]], out_name(key))
  write.csv(do.call(rbind, per_rep), out_file, row.names = FALSE)
  cat(sprintf("saved %s\n", out_file))
}
