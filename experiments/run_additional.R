## The eight methods of rlrchr/rlr_chr.R on the additional datasets, with WSC at slab masses 0.05, 0.10, 0.20.
## Usage: Rscript experiments/run_additional.R [--datasets=bike,blog --reps=1:100 --alpha=0.10]

arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(gsub("~\\+~", " ", sub("^--file=", "", arg[1L])))  # Rscript writes spaces as ~+~
source(file.path(here, "..", "rlrchr", "rlr_chr.R"))

flags <- parse_args(c(list(datasets = names(dataset_registry("additional")), reps = 1:100, n_workers = 5L),
                      default_config[c("alpha", "n_test", "bins", "levels")],
                      list(data = default_data_dir, out = default_out_dir)))
cfg <- modifyList(default_config, flags[c("alpha", "n_test", "bins", "levels")])

## additional/<key>_long.csv at the paper's alpha = 0.10, else ablations/alpha<100 alpha>_<key>_long.csv
out_name <- function(key) {
  if (cfg[["alpha"]] == 0.10) return(file.path("additional", paste0(key, "_long.csv")))
  file.path("ablations", sprintf("alpha%03d_%s_long.csv", as.integer(round(100 * cfg[["alpha"]])), key))
}

for (key in flags[["datasets"]]) {
  long <- run_dataset(key, flags[["reps"]], flags[["n_workers"]], cfg, wsc_deltas = c(0.05, 0.10, 0.20),
                      data_dir = flags[["data"]])
  ## alpha follows the dataset, so that files of different levels can be told apart
  long <- cbind(long[c("rep", "method", "dataset")], alpha = cfg[["alpha"]],
                long[setdiff(names(long), c("rep", "method", "dataset"))])
  out_file <- result_path(flags[["out"]], out_name(key))
  write.csv(long, out_file, row.names = FALSE)
  cat(sprintf("saved %s\n", out_file))
}
