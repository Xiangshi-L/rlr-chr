## RLR-CHR at several numbers T of levels, from one set of fits; the reference length is CHR at the default T.
## Usage: Rscript experiments/run_t_ablation.R --datasets=sgemm [--levels=25,50,100,200,1000 --bins=1000]

arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(gsub("~\\+~", " ", sub("^--file=", "", arg[1L])))  # Rscript writes spaces as ~+~
source(file.path(here, "..", "rlrchr", "rlr_chr.R"))

flags <- parse_args(c(list(datasets = NA_character_, reps = 1:100, n_workers = 4L),
                      default_config[c("alpha", "n_test", "bins")],
                      list(levels = c(25L, 50L, 100L, 200L, 1000L), data = default_data_dir,
                           out = default_out_dir)),
                    lists = c("datasets", "reps", "levels"))
cfg <- modifyList(default_config, flags[c("alpha", "n_test", "bins")])
reps <- flags[["reps"]]
levels_list <- sort(unique(flags[["levels"]]))

## Networks, histograms, Z and draws are fixed; T changes the levels of scores and intervals, and k.
t_rep <- function(rep_i) {
  alpha <- cfg[["alpha"]]
  d <- split_data(rep_i, ds$X, ds$Y, cfg[["n_test"]])
  q_chr <- fit_quantile_network(d, cfg)
  keras_gc()
  h_y <- chr_histograms(q_chr, bin_edges(d$Y_tr_std, cfg[["bins"]]))
  eps_ca_chr <- runif(d$m)
  eps_te_chr <- runif(length(d$Y_te))
  chr <- chr_intervals(d, h_y, level_grid(default_config[["levels"]], alpha), alpha, eps_ca_chr, eps_te_chr)
  chr_length <- mean(chr$hi - chr$lo)

  nuis <- fit_nuisances(d, rep_i, cfg)
  h_r <- chr_histograms(nuis$q, bin_edges(nuis$R_chr_tr, cfg[["bins"]]))
  Z <- localizer_coordinates(nuis$sig_ca, nuis$sig_te)
  eps_ca_r <- runif(d$m)
  eps_te_r <- runif(length(d$Y_te))
  wsc_setup <- wsc_prepare(d$X_te, seed = 1000 + rep_i)

  rows <- lapply(levels_list, function(n_levels) {
    grid <- level_grid(n_levels, alpha)
    E_cal <- chr_score_many(nuis$R_chr_ca, h_r$calib, h_r$edges, grid$tau_grid, grid$t0, eps_ca_r)
    k <- select_k(d$m, n_levels)$k
    rlr <- rlr_chr_intervals(E_cal, Z, k, h_r, grid, alpha, eps_te_r, nuis)
    covered <- d$Y_te >= rlr$lo & d$Y_te <= rlr$hi
    row <- data.frame(rep = rep_i, T = n_levels, k = k, coverage = mean(covered),
                      length = mean(rlr$hi - rlr$lo))
    for (delta in c(0.05, 0.10, 0.20)) row[[wsc_column(delta)]] <- wsc_from_setup(wsc_setup, covered, delta)
    row$chr_length <- chr_length
    row$rel_length <- row$length / chr_length
    row
  })
  keras_gc()
  do.call(rbind, rows)
}

for (key in flags[["datasets"]]) {
  ds <- read_dataset(key, flags[["data"]])
  cat(sprintf("\n%s: N = %d, p = %d; T = %s, %d bins, alpha = %.2f, repetitions %d-%d; %s\n",
              ds$name, nrow(ds$X), ncol(ds$X), paste(levels_list, collapse = ","), cfg[["bins"]],
              cfg[["alpha"]], min(reps), max(reps), workers_label(min(flags[["n_workers"]], length(reps)))))
  per_rep <- run_reps(reps, t_rep, flags[["n_workers"]], export = c("ds", "cfg", "levels_list"))
  out_file <- result_path(flags[["out"]], "ablations", paste0("t_", key, "_long.csv"))
  write.csv(do.call(rbind, per_rep), out_file, row.names = FALSE)
  cat(sprintf("saved %s\n", out_file))
}
