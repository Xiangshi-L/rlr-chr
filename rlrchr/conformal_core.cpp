// C++ core of RLR-CHR and RBC-CHR: CHR nested sets and scores, the kNN level selectors and WSC.
#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]

inline bool is_finite(double x) {
  return Rcpp::traits::is_finite<REALSXP>(x);
}

// Bin j in 1..n - 1 of x, with edges[j - 1] <= x < edges[j]; values outside
// [edges[0], edges[n - 1]) go to the first or the last bin.
inline int find_bin(double x, const NumericVector& edges) {
  int n = edges.size();
  if (n < 2) return 1;
  if (x <= edges[0]) return 1;
  if (x >= edges[n - 1]) return n - 1;
  int idx = std::upper_bound(edges.begin(), edges.end(), x) - edges.begin();
  if (idx < 1) idx = 1;
  if (idx > n - 1) idx = n - 1;
  return idx;
}

// Indices sorted by distance, ties broken by index.
inline std::vector<int> order_by_distance(const std::vector<double>& d) {
  int n = d.size();
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    if (d[a] < d[b]) return true;
    if (d[a] > d[b]) return false;
    return a < b;
  });
  return idx;
}

// CHR randomization: with probability (mass - tau) / min(pi_l, pi_u), drop the
// boundary bin of [l, u] with the smaller mass (eps is the uniform draw).
inline std::pair<int,int> chr_randomize(int l, int u, const NumericVector& pi_row, double tau,
                                        double eps) {
  double mass = 0.0;
  for (int i = l - 1; i <= u - 1; ++i) mass += pi_row[i];
  double min_end = std::min(pi_row[l - 1], pi_row[u - 1]);
  if (!is_finite(min_end) || min_end <= 0.0 || mass <= tau + 1e-12) return std::make_pair(l, u);
  double V = (mass - tau) / min_end;
  if (eps <= V) {
    if (pi_row[l - 1] <= pi_row[u - 1]) {
      if (l < u) l += 1;
    } else {
      if (u > l) u -= 1;
    }
  }
  return std::make_pair(l, u);
}

// Shortest bin interval [l, u] with mass >= tau inside [outer_l, outer_u] that,
// when has_inner, contains [inner_l, inner_u]. Ties in length go to the larger
// mass, matching the randomization, which drops the smaller boundary bin.
inline std::pair<int,int> shortest_bin_interval(const NumericVector& pi_row, double tau,
                                                int inner_l, int inner_u, bool has_inner,
                                                int outer_l, int outer_u) {
  int n_bins = pi_row.size();
  int L_min = std::max(1, outer_l);
  int U_max = std::min(n_bins, outer_u);
  if (L_min > U_max) return std::make_pair(1, n_bins);
  int L_req = has_inner ? inner_l : U_max;
  int U_req = has_inner ? inner_u : L_min;
  int l_lo = L_min;
  int l_hi = std::min(L_req, U_max);
  if (l_lo > l_hi) l_lo = l_hi;

  std::vector<double> cum_mass(n_bins + 1);
  cum_mass[0] = 0.0;
  for (int i = 1; i <= n_bins; ++i) cum_mass[i] = cum_mass[i - 1] + pi_row[i - 1];

  int best_l = L_min;
  int best_u = U_max;
  int best_len = n_bins + 1;
  double best_mass = -std::numeric_limits<double>::infinity();

  // For each left end, the first right end that reaches tau.
  for (int l = l_lo; l <= l_hi; ++l) {
    int u_start = std::max(l, U_req);
    if (u_start > U_max) continue;
    if ((cum_mass[U_max] - cum_mass[l - 1]) + 1e-12 < tau) continue;
    for (int u = u_start; u <= U_max; ++u) {
      double s = cum_mass[u] - cum_mass[l - 1];
      if (s + 1e-12 >= tau) {
        int len = u - l + 1;
        if (len < best_len || (len == best_len && s > best_mass)) {
          best_l = l;
          best_u = u;
          best_len = len;
          best_mass = s;
        }
        break;
      }
    }
  }
  return std::make_pair(best_l, best_u);
}

// Interval endpoints are bin edges, as in CHR; nesting of the index intervals
// carries over to the endpoints.
inline std::pair<double, double> chr_bin_endpoints(const NumericVector& edges, int l, int u) {
  return std::make_pair((double)edges[l - 1], (double)edges[u]);
}

// Window of at least a fraction delta of the sorted points with the lowest mean coverage;
// a longer window replaces the best one only if its mean is lower by more than 1e-12.
inline List worst_slab(const std::vector<double>& z_sorted,
                       const std::vector<double>& cov_sorted,
                       double delta) {
  int n = z_sorted.size();
  int len_min = std::max(1, (int)std::ceil(delta * n));
  std::vector<double> cum_cov(n + 1, 0.0);
  for (int i = 0; i < n; ++i) cum_cov[i + 1] = cum_cov[i] + cov_sorted[i];

  double best_mean = std::numeric_limits<double>::infinity();
  double best_a = NA_REAL;
  double best_b = NA_REAL;

  for (int len = len_min; len <= n; ++len) {
    int n_windows = n - len + 1;
    int best_j = -1;
    double local_best = std::numeric_limits<double>::infinity();
    for (int start = 0; start < n_windows; ++start) {
      int end = start + len - 1;
      double mean_cov = (cum_cov[end + 1] - cum_cov[start]) / len;
      if (mean_cov < local_best) {
        local_best = mean_cov;
        best_j = start;
      }
    }
    if (best_j >= 0 && is_finite(local_best) && (local_best + 1e-12 < best_mean)) {
      best_mean = local_best;
      best_a = z_sorted[best_j];
      best_b = z_sorted[best_j + len - 1];
    }
  }

  if (!is_finite(best_mean)) {
    return List::create(Named("mean_cov") = NA_REAL, Named("a") = NA_REAL, Named("b") = NA_REAL);
  }
  return List::create(Named("mean_cov") = best_mean, Named("a") = best_a, Named("b") = best_b);
}

// Nested CHR bin intervals S_1, ..., S_T (rows; columns l, u) for one
// histogram, built from level t0 upward and downward.
IntegerMatrix chr_build_nested_cpp(NumericVector pi_row, NumericVector tau_grid, int t0,
                                   double eps) {
  int n_levels = tau_grid.size();
  int n_bins = pi_row.size();
  IntegerMatrix S(n_levels, 2);
  IntegerMatrix S0(n_levels, 2);  // sets before randomization

  auto s0 = shortest_bin_interval(pi_row, tau_grid[t0 - 1], 0, 0, false, 1, n_bins);
  S0(t0 - 1, 0) = s0.first;
  S0(t0 - 1, 1) = s0.second;
  auto sr = chr_randomize(s0.first, s0.second, pi_row, tau_grid[t0 - 1], eps);
  S(t0 - 1, 0) = sr.first;
  S(t0 - 1, 1) = sr.second;

  // Upward: the set must contain S_{t-1}; the randomized set is kept only if
  // it still does.
  for (int t = t0 + 1; t <= n_levels; ++t) {
    int prev_l = S(t - 2, 0);
    int prev_u = S(t - 2, 1);
    auto s0_t = shortest_bin_interval(pi_row, tau_grid[t - 1], prev_l, prev_u, true, 1, n_bins);
    auto sr_t = chr_randomize(s0_t.first, s0_t.second, pi_row, tau_grid[t - 1], eps);
    if (sr_t.first <= prev_l && sr_t.second >= prev_u) {
      S(t - 1, 0) = sr_t.first;
      S(t - 1, 1) = sr_t.second;
    } else {
      S(t - 1, 0) = s0_t.first;
      S(t - 1, 1) = s0_t.second;
    }
  }

  // Downward: the set lies inside the unrandomized S_{t+1}; the randomized set
  // is then clipped into the randomized S_{t+1}, so the sequence is nested for
  // every histogram. An empty clip falls back to S_{t+1}.
  for (int t = t0 - 1; t >= 1; --t) {
    auto s0_t = shortest_bin_interval(pi_row, tau_grid[t - 1], 0, 0, false, S0(t, 0), S0(t, 1));
    S0(t - 1, 0) = s0_t.first;
    S0(t - 1, 1) = s0_t.second;
    auto sr_t = chr_randomize(s0_t.first, s0_t.second, pi_row, tau_grid[t - 1], eps);
    int l = std::max(sr_t.first, (int)S(t, 0));
    int u = std::min(sr_t.second, (int)S(t, 1));
    if (l > u) {
      l = S(t, 0);
      u = S(t, 1);
    }
    S(t - 1, 0) = l;
    S(t - 1, 1) = u;
  }

  return S;
}

// CHR histogram of each row of q_mat: bin masses of the piecewise-linear CDF through the
// quantiles, from 0 at the first edge to 1 at the last; a running maximum removes crossings.
// [[Rcpp::export]]
NumericMatrix quantiles_to_hist_probs_chr_cpp(NumericMatrix q_mat, NumericVector quantile_levels,
                                              NumericVector edges) {
  int n = q_mat.nrow();
  int n_q = q_mat.ncol();
  int n_bins = edges.size() - 1;
  if (n_bins < 1) stop("edges must have at least two values");
  if (quantile_levels.size() != n_q) stop("quantile_levels length must match ncol(q_mat)");

  // Knots (y_knot[h], p_knot[h]), h = 0, ..., n_q + 1, of the CDF.
  std::vector<double> p_knot(n_q + 2);
  p_knot[0] = 0.0;
  for (int h = 0; h < n_q; ++h) p_knot[h + 1] = quantile_levels[h];
  p_knot[n_q + 1] = 1.0;

  NumericMatrix out(n, n_bins);

  for (int i = 0; i < n; ++i) {
    std::vector<double> y_knot(n_q + 2);
    y_knot[0] = edges[0];
    for (int h = 0; h < n_q; ++h) y_knot[h + 1] = q_mat(i, h);
    y_knot[n_q + 1] = edges[n_bins];
    for (int h = 1; h < n_q + 2; ++h) {
      if (y_knot[h] < y_knot[h - 1]) y_knot[h] = y_knot[h - 1];
    }

    std::vector<double> row(n_bins, 0.0);
    for (int h = 0; h < n_q + 1; ++h) {
      double y1 = y_knot[h];
      double y2 = y_knot[h + 1];
      double mass = p_knot[h + 1] - p_knot[h];
      if (!is_finite(mass) || mass <= 0.0) continue;
      if (!is_finite(y1) || !is_finite(y2)) continue;
      if (y2 <= y1 + 1e-12) {
        row[find_bin(y1, edges) - 1] += mass;
      } else {
        // Spread the mass uniformly over [y1, y2].
        int j_start = find_bin(y1, edges);
        int j_end = find_bin(y2, edges);
        double seg_len = y2 - y1;
        for (int j = j_start; j <= j_end; ++j) {
          double lo = std::max(edges[j - 1], y1);
          double hi = std::min(edges[j], y2);
          if (hi > lo) row[j - 1] += mass * (hi - lo) / seg_len;
        }
      }
    }

    double s = 0.0;
    for (int j = 0; j < n_bins; ++j) s += row[j];
    if (!is_finite(s) || s <= 0.0) {
      double val = 1.0 / (double)n_bins;
      for (int j = 0; j < n_bins; ++j) out(i, j) = val;
    } else {
      for (int j = 0; j < n_bins; ++j) out(i, j) = row[j] / s;
    }
  }

  return out;
}

// CHR score of each point: the first level t with values[i] in [a_t, b_t],
// or T + 1 if there is none.
// [[Rcpp::export]]
IntegerVector chr_score_many_cpp(NumericVector values,
                                 NumericMatrix pi_mat,
                                 NumericVector edges,
                                 NumericVector tau_grid,
                                 int t0,
                                 NumericVector eps) {
  int n = pi_mat.nrow();
  int n_levels = tau_grid.size();
  IntegerVector out(n, n_levels + 1);
  if (values.size() != n) stop("values length must match nrow(pi_mat)");
  if (eps.size() != n) stop("eps length must match nrow(pi_mat)");

  for (int i = 0; i < n; ++i) {
    NumericVector pi_row = pi_mat(i, _);
    IntegerMatrix S = chr_build_nested_cpp(pi_row, tau_grid, t0, eps[i]);
    int E_i = n_levels + 1;
    double value = values[i];
    for (int t = 0; t < n_levels; ++t) {
      auto ab = chr_bin_endpoints(edges, S(t, 0), S(t, 1));
      if (value >= ab.first && value <= ab.second) {
        E_i = t + 1;
        break;
      }
    }
    out[i] = E_i;
  }
  return out;
}

// CHR interval [a, b] of each point at level t_use[i], clamped to 1..T.
// [[Rcpp::export]]
NumericMatrix chr_interval_many_cpp(NumericMatrix pi_mat,
                                    NumericVector edges,
                                    NumericVector tau_grid,
                                    int t0,
                                    NumericVector eps,
                                    IntegerVector t_use) {
  int n = pi_mat.nrow();
  int n_levels = tau_grid.size();
  if (eps.size() != n) stop("eps length must match nrow(pi_mat)");
  if (t_use.size() != n) stop("t_use length must match nrow(pi_mat)");
  NumericMatrix out(n, 2);

  for (int i = 0; i < n; ++i) {
    NumericVector pi_row = pi_mat(i, _);
    IntegerMatrix S = chr_build_nested_cpp(pi_row, tau_grid, t0, eps[i]);
    int t_i = t_use[i];
    if (t_i < 1) t_i = 1;
    if (t_i > n_levels) t_i = n_levels;
    auto ab = chr_bin_endpoints(edges, S(t_i - 1, 0), S(t_i - 1, 1));
    out(i, 0) = ab.first;
    out(i, 1) = ab.second;
  }
  return out;
}

// For each calibration point r: the distance to its k-th nearest neighbor in Z (r excluded,
// ties by index) and the score counts among its k and k - 1 neighbors; k is at most m - 1.
// [[Rcpp::export]]
List rbc_chr_precompute_cpp(IntegerVector E_cal, NumericVector Z_cal, int k) {
  int m = E_cal.size();
  int k_use = std::min(k, m - 1);
  if (m <= 1 || k_use <= 0) {
    return List::create(
      Named("Z_cal") = Z_cal,
      Named("kth_dist") = NumericVector(m, NA_REAL),
      Named("count_neighbors_k") = IntegerMatrix(m, 0),
      Named("count_neighbors_km1") = IntegerMatrix(m, 0),
      Named("k") = 0
    );
  }

  NumericVector kth_dist(m);
  int n_cols = max(E_cal) + 1;
  IntegerMatrix count_neighbors_k(m, n_cols);
  IntegerMatrix count_neighbors_km1(m, n_cols);

  for (int r = 0; r < m; ++r) {
    std::vector<double> d(m);
    for (int j = 0; j < m; ++j) d[j] = std::fabs(Z_cal[j] - Z_cal[r]);
    d[r] = R_PosInf;
    std::vector<int> ord = order_by_distance(d);
    std::vector<int> counts_k(n_cols, 0);
    std::vector<int> counts_km1(n_cols, 0);
    for (int j = 0; j < k_use; ++j) {
      int e = E_cal[ord[j]];
      if (e >= 1 && e <= n_cols) {
        counts_k[e - 1] += 1;
        if (j < k_use - 1) counts_km1[e - 1] += 1;
      }
    }
    kth_dist[r] = d[ord[k_use - 1]];
    for (int e = 0; e < n_cols; ++e) {
      count_neighbors_k(r, e) = counts_k[e];
      count_neighbors_km1(r, e) = counts_km1[e];
    }
  }

  return List::create(
    Named("Z_cal") = Z_cal,
    Named("kth_dist") = kth_dist,
    Named("count_neighbors_k") = count_neighbors_k,
    Named("count_neighbors_km1") = count_neighbors_km1,
    Named("k") = k_use
  );
}

// The G2 check at one p: it passes when the test point's selected quantile is the appended top
// value, or when A_1(p) >= 1 - alpha and A_2(p) >= 1 - alpha; level is the test level at p.
struct G2Check {
  bool pass;
  double level;
};

// Level at the smallest p = j / (m + 1), j = 1, ..., m + 1, that passes
// check(j), found by bisection on j; p = 1 always passes.
template <typename Check>
inline double smallest_passing_level(int m, Check check) {
  G2Check first = check(1);
  if (first.pass) return first.level;
  int lo = 1;
  int hi = m + 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (check(mid).pass) hi = mid;
    else lo = mid;
  }
  return check(hi).level;
}

// RBC-CHR level at one test point: the G2 rule on the integer CHR scores E_cal,
// with the neighbor counts of rbc_chr_precompute_cpp; +Inf when the selected quantile
// is the appended top value or there are no neighbors (rbc_chr_intervals maps it to level T).
// Each check scans the count tables, O(m T); g2_residual_knn_predict_one_cpp evaluates the same
// rule with sorted neighbor scores in O(m log(k + 1)) worst-case time per check and returns
// the same level on integer scores.
double rbc_chr_g2_knn_predict_one_cpp(IntegerVector E_cal,
                                     NumericVector Z_cal,
                                     NumericVector kth_dist,
                                     IntegerMatrix count_neighbors_k,
                                     IntegerMatrix count_neighbors_km1,
                                     int k,
                                     double Z_test,
                                     double alpha) {
  int m = E_cal.size();
  int n_cols = count_neighbors_k.ncol();
  if (m == 0 || k <= 0 || n_cols == 0) return R_PosInf;

  std::vector<double> d_test(m);
  for (int i = 0; i < m; ++i) d_test[i] = std::fabs(Z_cal[i] - Z_test);
  std::vector<int> ord = order_by_distance(d_test);

  // Score counts among the k nearest calibration points of the test point.
  std::vector<int> count_test(n_cols, 0);
  for (int i = 0; i < k; ++i) {
    int e = E_cal[ord[i]];
    if (e >= 1 && e <= n_cols) count_test[e - 1] += 1;
  }

  std::vector<int> cum_test(n_cols, 0);
  int cum_count = 0;
  for (int e = 0; e < n_cols; ++e) {
    cum_count += count_test[e];
    cum_test[e] = cum_count;
  }

  // The test point enters N_k(r) when it is closer to r than r's k-th neighbor.
  std::vector<int> in_nbhd(m, 0);
  for (int r = 0; r < m; ++r) in_nbhd[r] = (d_test[r] < kth_dist[r]) ? 1 : 0;

  auto check = [&](int j) -> G2Check {
    double p = (double)j / (m + 1.0);
    int thr_test = (int)std::ceil(p * (k + 1.0));
    if (thr_test > k) return {true, R_PosInf};

    int idx_test = -1;
    for (int e = 0; e < n_cols; ++e) {
      if (cum_test[e] >= thr_test) {
        idx_test = e + 1;
        break;
      }
    }
    int t_test = (idx_test == -1) ? n_cols : idx_test;

    long count_A1 = 0;
    long count_A2 = 0;
    std::vector<int> base_count(n_cols);
    std::vector<int> count1(n_cols);

    for (int r = 0; r < m; ++r) {
      // Scores in N_k(r) plus r itself, without the test point.
      const IntegerMatrix& base_mat = in_nbhd[r] ? count_neighbors_km1 : count_neighbors_k;
      int sum_base = 0;
      for (int e = 0; e < n_cols; ++e) {
        int val = base_mat(r, e);
        base_count[e] = val;
        sum_base += val;
      }
      int E_r = E_cal[r];
      if (E_r >= 1 && E_r <= n_cols) {
        base_count[E_r - 1] += 1;
        sum_base += 1;
      }

      int n_r = sum_base + (in_nbhd[r] ? 1 : 0);
      int thr_r = (int)std::ceil(p * (double)n_r);

      // v_{r,1}: the test point enters with score t_test.
      for (int e = 0; e < n_cols; ++e) count1[e] = base_count[e];
      if (in_nbhd[r] && t_test >= 1 && t_test <= n_cols) {
        count1[t_test - 1] += 1;
      }

      int cum1 = 0;
      int v_r1 = n_cols;
      for (int e = 0; e < n_cols; ++e) {
        cum1 += count1[e];
        if (cum1 >= thr_r) {
          v_r1 = e + 1;
          break;
        }
      }

      // v_{r,2}: the test point enters with score 0, below every level.
      int v_r2 = n_cols;
      if (in_nbhd[r]) {
        if (thr_r == 1) {
          v_r2 = 0;
        } else {
          int target = thr_r - 1;
          int cum2 = 0;
          for (int e = 0; e < n_cols; ++e) {
            cum2 += base_count[e];
            if (cum2 >= target) {
              v_r2 = e + 1;
              break;
            }
          }
        }
      } else {
        int cum2 = 0;
        for (int e = 0; e < n_cols; ++e) {
          cum2 += base_count[e];
          if (cum2 >= thr_r) {
            v_r2 = e + 1;
            break;
          }
        }
      }

      if (E_r <= v_r1) count_A1 += 1;
      if (E_r <= v_r2) count_A2 += 1;
    }

    double A_1 = (double)count_A1 / (m + 1.0);
    double A_2 = ((double)count_A2 + 1.0) / (m + 1.0);
    return {(A_1 >= 1.0 - alpha && A_2 >= 1.0 - alpha), (double)t_test};
  };

  return smallest_passing_level(m, check);
}

// RBC-CHR level at each test point (see rbc_chr_g2_knn_predict_one_cpp).
// [[Rcpp::export]]
NumericVector rbc_chr_g2_knn_predict_many_cpp(IntegerVector E_cal,
                                              NumericVector Z_cal,
                                              NumericVector kth_dist,
                                              IntegerMatrix count_neighbors_k,
                                              IntegerMatrix count_neighbors_km1,
                                              int k,
                                              NumericVector Z_test,
                                              double alpha) {
  int n = Z_test.size();
  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    out[i] = rbc_chr_g2_knn_predict_one_cpp(
      E_cal, Z_cal, kth_dist, count_neighbors_k, count_neighbors_km1, k, Z_test[i], alpha
    );
  }
  return out;
}

// RLR-CHR level at each test point: e* = max{e : p(e) > alpha}, or 1 if no
// level passes, where p(e) compares the test point's local rank A_{n+1}(e)
// with the calibration ranks A_i(e). rank_base[i] is A_i without the test point.
// [[Rcpp::export]]
IntegerVector rlr_chr_select_threshold_many_cpp(IntegerVector E_cal,
                                                NumericVector Z_cal,
                                                NumericVector kth_dist,
                                                IntegerVector kth_score,
                                                IntegerVector rank_base,
                                                int k,
                                                int n_levels,
                                                NumericVector Z_test,
                                                double alpha) {
  int m = E_cal.size();
  int n_test = Z_test.size();
  IntegerVector out(n_test, 1);
  if (m == 0 || k <= 0 || n_levels <= 0) return out;

  std::vector<double> d_test(m);
  std::vector<int> in_nbhd(m, 0);
  std::vector<int> kth_displaced(m, 0);

  for (int i_test = 0; i_test < n_test; ++i_test) {
    double z_test = Z_test[i_test];
    for (int i = 0; i < m; ++i) d_test[i] = std::fabs(Z_cal[i] - z_test);
    std::vector<int> ord = order_by_distance(d_test);

    // When the test point enters N_k(i) it displaces i's k-th neighbor, whose
    // contribution to the rank is removed.
    for (int i = 0; i < m; ++i) {
      in_nbhd[i] = (d_test[i] < kth_dist[i]) ? 1 : 0;
      kth_displaced[i] = (in_nbhd[i] && kth_score[i] <= E_cal[i]) ? 1 : 0;
    }

    // p(e) is nonincreasing in e, so scan e downward from T.
    int e_star = 1;
    for (int e = n_levels; e >= 1; --e) {
      int A_test = 1;
      for (int j = 0; j < k; ++j) {
        if (E_cal[ord[j]] <= e) A_test += 1;
      }

      int ge_count = 1;
      for (int i = 0; i < m; ++i) {
        int A_i = rank_base[i] - kth_displaced[i] + (in_nbhd[i] && e <= E_cal[i] ? 1 : 0);
        if (A_i >= A_test) ge_count += 1;
      }

      double p_e = (double)ge_count / (m + 1.0);
      if (p_e > alpha) {
        e_star = e;
        break;
      }
    }
    out[i_test] = e_star;
  }

  return out;
}

// G2-Residual threshold at one test point: the G2 rule on the continuous scores E_cal, with
// the sorted scores of each calibration point's k and k - 1 neighbors; +Inf when the selected
// quantile is the appended +Inf, an unbounded interval.
// [[Rcpp::export]]
double g2_residual_knn_predict_one_cpp(NumericVector E_cal,
                                       NumericVector Z_cal,
                                       NumericVector kth_dist,
                                       NumericMatrix sorted_E_k,
                                       NumericMatrix sorted_E_km1,
                                       int k,
                                       double Z_test,
                                       double alpha) {
  int m = E_cal.size();

  std::vector<double> d_test(m);
  for (int i = 0; i < m; ++i) d_test[i] = std::fabs(Z_cal[i] - Z_test);

  std::vector<int> ord(m);
  for (int i = 0; i < m; ++i) ord[i] = i;
  std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
    [&](int a, int b) {
      if (d_test[a] < d_test[b]) return true;
      if (d_test[a] > d_test[b]) return false;
      return a < b;
    });

  std::vector<double> E_test_sorted(k);
  for (int i = 0; i < k; ++i) E_test_sorted[i] = E_cal[ord[i]];
  std::sort(E_test_sorted.begin(), E_test_sorted.end());

  std::vector<bool> in_nbhd(m);
  for (int r = 0; r < m; ++r) in_nbhd[r] = (d_test[r] < kth_dist[r]);

  // Sorted scores of N_k(r) plus r itself, without the test point.
  int km1 = k - 1;
  std::vector<std::vector<double>> base_sorted(m);
  for (int r = 0; r < m; ++r) {
    double E_r = E_cal[r];
    if (in_nbhd[r] && k > 1) {
      std::vector<double> tmp(km1);
      for (int j = 0; j < km1; ++j) tmp[j] = sorted_E_km1(r, j);
      base_sorted[r].resize(km1 + 1);
      auto it = std::upper_bound(tmp.begin(), tmp.end(), E_r);
      int pos = (int)(it - tmp.begin());
      for (int j = 0; j < pos; ++j) base_sorted[r][j] = tmp[j];
      base_sorted[r][pos] = E_r;
      for (int j = pos; j < km1; ++j) base_sorted[r][j + 1] = tmp[j];
    } else if (in_nbhd[r] && k == 1) {
      base_sorted[r] = {E_r};
    } else {
      std::vector<double> tmp(k);
      for (int j = 0; j < k; ++j) tmp[j] = sorted_E_k(r, j);
      base_sorted[r].resize(k + 1);
      auto it = std::upper_bound(tmp.begin(), tmp.end(), E_r);
      int pos = (int)(it - tmp.begin());
      for (int j = 0; j < pos; ++j) base_sorted[r][j] = tmp[j];
      base_sorted[r][pos] = E_r;
      for (int j = pos; j < k; ++j) base_sorted[r][j + 1] = tmp[j];
    }
  }

  // Left p-quantile of the sorted values with one value added.
  auto quantile_with = [](const std::vector<double>& sarr, double val, double p) -> double {
    int n = (int)sarr.size() + 1;
    int thr = (int)std::ceil(p * (double)n);
    if (thr > n) return R_PosInf;
    auto it = std::upper_bound(sarr.begin(), sarr.end(), val);
    int pos = (int)(it - sarr.begin()) + 1;
    if (thr < pos) return sarr[thr - 1];
    else if (thr == pos) return val;
    else return sarr[thr - 2];
  };

  auto check = [&](int j) -> G2Check {
    double p = (double)j / (m + 1.0);
    int thr_test = (int)std::ceil(p * (k + 1.0));
    if (thr_test > k) return {true, R_PosInf};

    double t_test = E_test_sorted[thr_test - 1];
    if (!is_finite(t_test)) return {true, R_PosInf};

    // v_{r,1} and v_{r,2} add the test score t_test and 0 when the test point
    // is in N_k(r).
    long count_A1 = 0;
    long count_A2 = 0;
    for (int r = 0; r < m; ++r) {
      double E_r = E_cal[r];
      const std::vector<double>& bs = base_sorted[r];
      double v_r1;
      double v_r2;
      if (in_nbhd[r]) {
        v_r1 = quantile_with(bs, t_test, p);
        v_r2 = quantile_with(bs, 0.0, p);
      } else {
        int n_r = (int)bs.size();
        int thr_r = (int)std::ceil(p * (double)n_r);
        v_r1 = (thr_r <= n_r) ? bs[thr_r - 1] : R_PosInf;
        v_r2 = v_r1;
      }
      if (E_r <= v_r1) ++count_A1;
      if (E_r <= v_r2) ++count_A2;
    }

    double A_1 = (double)count_A1 / (m + 1.0);
    double A_2 = ((double)count_A2 + 1.0) / (m + 1.0);
    return {(A_1 >= 1.0 - alpha && A_2 >= 1.0 - alpha), t_test};
  };

  return smallest_passing_level(m, check);
}

// Worst-slab coverage: over the directions (columns of Z_est), the slab of at
// least a fraction delta of the points with the lowest mean of covered01.
// [[Rcpp::export]]
List wsc_find_best_direction_cpp(NumericMatrix Z_est, NumericVector covered01, double delta) {
  int n = Z_est.nrow();
  int B = Z_est.ncol();
  if (covered01.size() != n || n == 0 || B == 0) {
    return List::create(Named("mean_cov") = NA_REAL, Named("dir_idx") = NA_INTEGER,
                        Named("a") = NA_REAL, Named("b") = NA_REAL);
  }

  double best_mean = std::numeric_limits<double>::infinity();
  int best_dir_idx = NA_INTEGER;
  double best_a = NA_REAL;
  double best_b = NA_REAL;
  std::vector<int> ord(n);
  std::vector<double> z_sorted(n);
  std::vector<double> cov_sorted(n);

  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < n; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int c) {
      if (Z_est(a, b) < Z_est(c, b)) return true;
      if (Z_est(a, b) > Z_est(c, b)) return false;
      return a < c;
    });

    for (int i = 0; i < n; ++i) {
      z_sorted[i] = Z_est(ord[i], b);
      cov_sorted[i] = covered01[ord[i]];
    }

    List slab = worst_slab(z_sorted, cov_sorted, delta);
    double mean_cov = as<double>(slab["mean_cov"]);
    if (is_finite(mean_cov) && mean_cov < best_mean) {
      best_mean = mean_cov;
      best_dir_idx = b + 1;
      best_a = as<double>(slab["a"]);
      best_b = as<double>(slab["b"]);
    }
  }

  return List::create(Named("mean_cov") = best_mean, Named("dir_idx") = best_dir_idx,
                      Named("a") = best_a, Named("b") = best_b);
}
