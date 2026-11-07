# Physical PPDF retrieval, round 3 (2018-07-01 + slab MC)

## Slab MC (error of g-hat, ppm; no training; matched: ['o2a_f_am', 'o2a_k1_am', 'o2a_k2diag_am', 'sco2_k1_am', 'sco2_k1w_am', 'sco2_k2diag_am', 'wco2_k1_am', 'wco2_k1w_am'])

| noise      | subset                      |    n |   conv |   rsd_ppm |   sd_ppm |   bias_ppm |   med_bias_ppm |   corr_ghat_g |   med_dof_g |   med_sig_g_ppm |   med_chi2_obs |   med_ms |
|:-----------|:----------------------------|-----:|-------:|----------:|---------:|-----------:|---------------:|--------------:|------------:|----------------:|---------------:|---------:|
| noise-free | all                         | 2000 |  0.976 |     1.587 |    9.475 |      0.7   |          0.992 |         0.189 |       0.904 |           1.238 |          1.951 |  201.188 |
| noise-free | clear (no cloud, tau_a<0.1) |  852 |  1     |     0.667 |    1.572 |      0.869 |          0.522 |         0.792 |       0.93  |           1.057 |          0.68  |  166.549 |
| noise-free | cloudy                      |  608 |  0.926 |     7.459 |   16.68  |     -1.028 |          2.826 |         0.085 |       0.823 |           1.684 |         19.111 |  299.62  |
| noise-free | land                        | 1185 |  0.986 |     1.435 |    7.65  |      0.858 |          1.038 |         0.273 |       0.922 |           1.117 |          2.067 |  175.435 |
| noise-free | ocean                       |  815 |  0.961 |     1.885 |   11.624 |      0.47  |          0.939 |         0.112 |       0.862 |           1.485 |          1.813 |  220.121 |
| noise-free | clear land                  |  512 |  1     |     0.632 |    1.539 |      0.807 |          0.533 |         0.808 |       0.936 |           1.014 |          0.633 |  158.948 |
| noise-free | clear ocean                 |  340 |  1     |     0.731 |    1.614 |      0.961 |          0.477 |         0.768 |       0.913 |           1.178 |          0.695 |  202.06  |
| noisy      | all                         | 2000 |  0.974 |     1.675 |    9.444 |      0.703 |          1.04  |         0.189 |       0.905 |           1.235 |          2.209 |  201.054 |
| noisy      | clear (no cloud, tau_a<0.1) |  852 |  1     |     0.738 |    1.625 |      0.896 |          0.559 |         0.783 |       0.93  |           1.058 |          0.958 |  166.264 |
| noisy      | cloudy                      |  608 |  0.923 |     7.285 |   16.607 |     -1.012 |          2.921 |         0.087 |       0.822 |           1.688 |         18.955 |  295.231 |
| noisy      | land                        | 1185 |  0.985 |     1.476 |    7.384 |      0.885 |          1.065 |         0.284 |       0.921 |           1.122 |          2.371 |  179.627 |
| noisy      | ocean                       |  815 |  0.957 |     1.984 |   11.809 |      0.44  |          1.009 |         0.108 |       0.861 |           1.489 |          1.983 |  221.741 |
| noisy      | clear land                  |  512 |  1     |     0.681 |    1.574 |      0.829 |          0.557 |         0.802 |       0.936 |           1.012 |          0.998 |  159.966 |
| noisy      | clear ocean                 |  340 |  1     |     0.853 |    1.694 |      0.997 |          0.564 |         0.752 |       0.912 |           1.184 |          0.912 |  198.307 |

pull rsd / median (noise-free):

|                       |   rsd |   median |
|:----------------------|------:|---------:|
| r_pull_o2a_k1_am      | 0.446 |    0.273 |
| r_pull_o2a_f_am       | 0.175 |    0.047 |
| r_pull_o2a_k2diag_am  | 0.55  |    0.351 |
| r_pull_wco2_k1_am     | 0.735 |    0.402 |
| r_pull_wco2_k1w_am    | 0.207 |    0.182 |
| r_pull_sco2_k1_am     | 0.737 |   -0.396 |
| r_pull_sco2_k1w_am    | 0.173 |   -0.156 |
| r_pull_sco2_k2diag_am | 0.385 |    0.167 |

corr(retrieved ln tau_L, true ln tau_a) [clean]: 0.28; corr(sigma_H, true layer mid sigma): 0.49

pull rsd / median (noisy):

|                         |   rsd |   median |
|:------------------------|------:|---------:|
| r_n_pull_o2a_k1_am      | 0.463 |    0.268 |
| r_n_pull_o2a_f_am       | 0.207 |    0.054 |
| r_n_pull_o2a_k2diag_am  | 0.623 |    0.298 |
| r_n_pull_wco2_k1_am     | 0.836 |    0.456 |
| r_n_pull_wco2_k1w_am    | 0.252 |    0.187 |
| r_n_pull_sco2_k1_am     | 0.849 |   -0.454 |
| r_n_pull_sco2_k1w_am    | 0.211 |   -0.157 |
| r_n_pull_sco2_k2diag_am | 0.479 |    0.171 |

corr(retrieved ln tau_L, true ln tau_a) [n_]: 0.28; corr(sigma_H, true layer mid sigma): 0.49

## Slab MC_nof (error of g-hat, ppm; no training; matched: ['o2a_k1_am', 'o2a_k2diag_am', 'sco2_k1_am', 'sco2_k1w_am', 'sco2_k2diag_am', 'wco2_k1_am', 'wco2_k1w_am'])

| noise      | subset                      |    n |   conv |   rsd_ppm |   sd_ppm |   bias_ppm |   med_bias_ppm |   corr_ghat_g |   med_dof_g |   med_sig_g_ppm |   med_chi2_obs |   med_ms |
|:-----------|:----------------------------|-----:|-------:|----------:|---------:|-----------:|---------------:|--------------:|------------:|----------------:|---------------:|---------:|
| noise-free | all                         | 2000 |  0.97  |     1.622 |    9.461 |      0.517 |          1     |         0.182 |       0.896 |           1.291 |          1.631 |  198.128 |
| noise-free | clear (no cloud, tau_a<0.1) |  852 |  1     |     0.699 |    1.651 |      0.939 |          0.537 |         0.773 |       0.926 |           1.091 |          0.65  |  163.285 |
| noise-free | cloudy                      |  608 |  0.908 |     7.722 |   16.519 |     -1.713 |          2.778 |         0.072 |       0.806 |           1.764 |         13.449 |  297.829 |
| noise-free | land                        | 1185 |  0.981 |     1.411 |    7.66  |      0.837 |          1.01  |         0.266 |       0.918 |           1.149 |          1.864 |  170.814 |
| noise-free | ocean                       |  815 |  0.955 |     2.074 |   11.574 |      0.051 |          0.975 |         0.104 |       0.842 |           1.59  |          1.447 |  217.897 |
| noise-free | clear land                  |  512 |  1     |     0.651 |    1.469 |      0.828 |          0.54  |         0.818 |       0.934 |           1.03  |          0.617 |  155.404 |
| noise-free | clear ocean                 |  340 |  1     |     0.81  |    1.881 |      1.107 |          0.522 |         0.707 |       0.902 |           1.253 |          0.695 |  197.643 |
| noisy      | all                         | 2000 |  0.972 |     1.725 |    9.511 |      0.54  |          1.053 |         0.182 |       0.896 |           1.292 |          1.847 |  197.541 |
| noisy      | clear (no cloud, tau_a<0.1) |  852 |  1     |     0.767 |    1.697 |      0.963 |          0.569 |         0.764 |       0.926 |           1.09  |          0.905 |  162.404 |
| noisy      | cloudy                      |  608 |  0.913 |     7.32  |   16.621 |     -1.663 |          2.954 |         0.073 |       0.797 |           1.804 |         13.786 |  307.42  |
| noisy      | land                        | 1185 |  0.983 |     1.503 |    7.77  |      0.796 |          1.043 |         0.263 |       0.917 |           1.151 |          2.162 |  174.412 |
| noisy      | ocean                       |  815 |  0.957 |     2.214 |   11.574 |      0.169 |          1.056 |         0.107 |       0.841 |           1.593 |          1.573 |  216.663 |
| noisy      | clear land                  |  512 |  1     |     0.695 |    1.504 |      0.85  |          0.563 |         0.812 |       0.934 |           1.028 |          0.955 |  157.133 |
| noisy      | clear ocean                 |  340 |  1     |     0.926 |    1.941 |      1.132 |          0.585 |         0.694 |       0.901 |           1.257 |          0.839 |  193.631 |

pull rsd / median (noise-free):

|                       |   rsd |   median |
|:----------------------|------:|---------:|
| r_pull_o2a_k1_am      | 0.437 |    0.245 |
| r_pull_o2a_k2diag_am  | 0.611 |    0.383 |
| r_pull_wco2_k1_am     | 0.634 |    0.336 |
| r_pull_wco2_k1w_am    | 0.183 |    0.143 |
| r_pull_sco2_k1_am     | 0.596 |   -0.309 |
| r_pull_sco2_k1w_am    | 0.171 |   -0.139 |
| r_pull_sco2_k2diag_am | 0.345 |    0.21  |

corr(retrieved ln tau_L, true ln tau_a) [clean]: 0.27; corr(sigma_H, true layer mid sigma): 0.44

pull rsd / median (noisy):

|                         |   rsd |   median |
|:------------------------|------:|---------:|
| r_n_pull_o2a_k1_am      | 0.45  |    0.231 |
| r_n_pull_o2a_k2diag_am  | 0.678 |    0.33  |
| r_n_pull_wco2_k1_am     | 0.739 |    0.402 |
| r_n_pull_wco2_k1w_am    | 0.224 |    0.148 |
| r_n_pull_sco2_k1_am     | 0.681 |   -0.348 |
| r_n_pull_sco2_k1w_am    | 0.213 |   -0.144 |
| r_n_pull_sco2_k2diag_am | 0.43  |    0.209 |

corr(retrieved ln tau_L, true ln tau_a) [n_]: 0.28; corr(sigma_H, true layer mid sigma): 0.44

## Real far-cloud QF0 soundings, 2018-07-01 (matched: ['o2a_k1_am', 'o2a_f_am', 'o2a_k2diag_am', 'wco2_k1_am', 'wco2_k1w_am', 'sco2_k1_am', 'sco2_k1w_am', 'sco2_k2diag_am']; round-2 fold 2)

| surface   | estimator                                   |    n |   rsd_vs_bc |   bias_vs_bc |   total |   rsd_vs_prior |   rsd_bc_vs_prior |     R2 |   corr_bcdev |
|:----------|:--------------------------------------------|-----:|------------:|-------------:|--------:|---------------:|------------------:|-------:|-------------:|
| ocean     | prior only                                  | 1000 |       0.92  |        0.189 |   0.939 |          0     |              0.92 |  0     |      nan     |
| ocean     | PHYS (raw, uncalibrated)                    | 1000 |       0.929 |       -2.802 |   2.952 |          1.057 |              0.92 | -0.021 |        0.719 |
| ocean     | PHYS + same-date offset [diagnostic only]   | 1000 |       0.929 |        0     |   0.929 |          1.057 |              0.92 | -0.021 |        0.719 |
| ocean     | PHYS, sigma_a(g)=0.25 % (analytic re-prior) | 1000 |       0.887 |       -1.182 |   1.478 |          0.523 |              0.92 |  0.071 |        0.677 |
| ocean     | E0_am (round 1/2)                           | 1000 |       2.391 |        0.491 |   2.441 |          2.439 |              0.92 | -5.758 |        0.409 |
| ocean     | E2cP_am_hgb (round 2, ML)                   | 1000 |       0.792 |       -0.11  |   0.799 |          0.982 |              0.92 |  0.259 |        0.786 |
| ocean     | OB_E2cP_am_hgb (round 2, ML + blend)        | 1000 |       0.671 |        0.15  |   0.687 |          0.765 |              0.92 |  0.468 |        0.786 |
| land      | prior only                                  | 1000 |       1.59  |        1.526 |   2.204 |          0     |              1.59 |  0     |      nan     |
| land      | PHYS (raw, uncalibrated)                    | 1000 |       1.535 |       -3.868 |   4.162 |          2.609 |              1.59 |  0.068 |        0.801 |
| land      | PHYS + same-date offset [diagnostic only]   | 1000 |       1.535 |       -0     |   1.535 |          2.609 |              1.59 |  0.068 |        0.801 |
| land      | PHYS, sigma_a(g)=0.25 % (analytic re-prior) | 1000 |       1.011 |       -0.844 |   1.317 |          1.203 |              1.59 |  0.595 |        0.773 |
| land      | E0_am (round 1/2)                           | 1000 |       2.014 |        1.852 |   2.736 |          2.813 |              1.59 | -0.605 |        0.675 |
| land      | E2cP_am_hgb (round 2, ML)                   | 1000 |       1.28  |        0.858 |   1.541 |          1.3   |              1.59 |  0.352 |        0.542 |
| land      | OB_E2cP_am_hgb (round 2, ML + blend)        | 1000 |       1.241 |        0.608 |   1.382 |          1.018 |              1.59 |  0.391 |        0.542 |

### Diagnostics

|                                 | 0       | 1       |
|:--------------------------------|:--------|:--------|
| surface                         | ocean   | land    |
| n                               | 1000    | 1000    |
| conv_rate                       | 1.0     | 1.0     |
| med_iter                        | 3.0     | 4.0     |
| med_nfwd                        | 30.0    | 36.0    |
| chi2_med                        | 160.654 | 63.556  |
| chi2_q90                        | 274.564 | 99.137  |
| chi2_per_obs_med                | 20.082  | 7.945   |
| dof_g_med                       | 0.928   | 0.921   |
| dof_g_q10                       | 0.891   | 0.883   |
| dof_g_q90                       | 0.936   | 0.936   |
| dof_total_med                   | 2.775   | 2.721   |
| sig_g_ppm_med                   | 1.071   | 1.128   |
| same_date_offset_ppm            | 2.802   | 3.868   |
| ms_med                          | 213.259 | 261.985 |
| ms_mean                         | 258.922 | 278.821 |
| tauL_med                        | 0.073   | 0.078   |
| sigmaH_med                      | 0.988   | 0.942   |
| alpha_med                       | 0.958   | 1.02    |
| h_med                           | 1.0     | 0.956   |
| s_p_med                         | 1.001   | 1.001   |
| g_med                           | 0.993   | 0.988   |
| frac_tauL_at_bound              | 0.0     | 0.0     |
| corr_tauL_aod_total             | 0.134   | 0.616   |
| spearman_tauL_aod               | 0.019   | 0.58    |
| spearman_sigmaH_dust_height     | -0.115  | -0.016  |
| spearman_h_h2o_scale            | 0.977   | 0.993   |
| corr_sp_psfc_ratio              | 0.05    | 0.332   |
| corr_g_minus1_vs_bc_minus_prior | 0.716   | 0.8     |

### Observables, ocean (medians)

|                |     obs |   F(prior state) |   resid (obs-F) med |   resid rsd |   pull rsd |   obs rsd |
|:---------------|--------:|-----------------:|--------------------:|------------:|-----------:|----------:|
| o2a_k1_am      |  0.9756 |           0.9587 |              0.0035 |      0.0072 |     1.6402 |    0.0098 |
| o2a_f_am       |  0.0126 |           0.0376 |             -0.0247 |      0.0056 |     2.8147 |    0.0024 |
| o2a_k2diag_am  | -0.0015 |           0.0111 |             -0.008  |      0.0082 |     1.3903 |    0.005  |
| wco2_k1_am     |  0.9879 |           0.99   |              0.0005 |      0.0018 |     0.7432 |    0.0029 |
| wco2_k1w_am    |  1.0059 |           0.9851 |              0.0104 |      0.0093 |     0.2863 |    0.0604 |
| sco2_k1_am     |  0.9992 |           1.003  |              0.0002 |      0.0035 |     1.2159 |    0.0061 |
| sco2_k1w_am    |  0.9997 |           0.991  |             -0.0057 |      0.0074 |     0.2462 |    0.0533 |
| sco2_k2diag_am |  0.0113 |           0.0122 |              0.0015 |      0.0044 |     1.3409 |    0.0052 |

### Observables, land (medians)

|                |     obs |   F(prior state) |   resid (obs-F) med |   resid rsd |   pull rsd |   obs rsd |
|:---------------|--------:|-----------------:|--------------------:|------------:|-----------:|----------:|
| o2a_k1_am      |  0.9914 |           0.9855 |              0.0055 |      0.0044 |     0.9977 |    0.0108 |
| o2a_f_am       |  0.0084 |           0.0227 |             -0.014  |      0.0038 |     1.8604 |    0.0017 |
| o2a_k2diag_am  | -0.0046 |           0.0068 |             -0.0107 |      0.0053 |     0.8137 |    0.0062 |
| wco2_k1_am     |  0.9837 |           0.9971 |             -0.002  |      0.0019 |     0.6793 |    0.0103 |
| wco2_k1w_am    |  0.9758 |           0.9997 |              0.0178 |      0.0143 |     0.3953 |    0.1072 |
| sco2_k1_am     |  0.9937 |           1.0028 |              0.0006 |      0.0028 |     0.9221 |    0.0066 |
| sco2_k1w_am    |  0.946  |           1.0011 |             -0.0142 |      0.0089 |     0.29   |    0.0924 |
| sco2_k2diag_am |  0.0057 |           0.0081 |             -0.0018 |      0.0059 |     1.7433 |    0.0066 |

Reconstructed formal sigma / stored sig_* (median):

|                      |     0 |
|:---------------------|------:|
| sigchk_sco2_k1       | 1     |
| sigchk_o2a_k1        | 1     |
| sigchk_wco2_k1       | 1     |
| sigchk_o2a_f         | 1     |
| corr_formal_o2a_k1_f | 0.713 |

## Real far-cloud QF0 soundings, 2018-07-01_nof (matched: ['o2a_k1_am', 'o2a_k2diag_am', 'wco2_k1_am', 'wco2_k1w_am', 'sco2_k1_am', 'sco2_k1w_am', 'sco2_k2diag_am']; round-2 fold 2)

| surface   | estimator                                   |    n |   rsd_vs_bc |   bias_vs_bc |   total |   rsd_vs_prior |   rsd_bc_vs_prior |     R2 |   corr_bcdev |
|:----------|:--------------------------------------------|-----:|------------:|-------------:|--------:|---------------:|------------------:|-------:|-------------:|
| ocean     | prior only                                  | 1000 |       0.92  |        0.189 |   0.939 |          0     |              0.92 |  0     |      nan     |
| ocean     | PHYS (raw, uncalibrated)                    | 1000 |       1.091 |       -2.582 |   2.803 |          1.115 |              0.92 | -0.407 |        0.702 |
| ocean     | PHYS + same-date offset [diagnostic only]   | 1000 |       1.091 |       -0     |   1.091 |          1.115 |              0.92 | -0.407 |        0.702 |
| ocean     | PHYS, sigma_a(g)=0.25 % (analytic re-prior) | 1000 |       0.978 |       -1.08  |   1.457 |          0.546 |              0.92 | -0.131 |        0.641 |
| ocean     | E0_am (round 1/2)                           | 1000 |       2.391 |        0.491 |   2.441 |          2.439 |              0.92 | -5.758 |        0.409 |
| ocean     | E2cP_am_hgb (round 2, ML)                   | 1000 |       0.792 |       -0.11  |   0.799 |          0.982 |              0.92 |  0.259 |        0.786 |
| ocean     | OB_E2cP_am_hgb (round 2, ML + blend)        | 1000 |       0.671 |        0.15  |   0.687 |          0.765 |              0.92 |  0.468 |        0.786 |
| land      | prior only                                  | 1000 |       1.59  |        1.526 |   2.204 |          0     |              1.59 |  0     |      nan     |
| land      | PHYS (raw, uncalibrated)                    | 1000 |       1.508 |       -3.802 |   4.09  |          2.51  |              1.59 |  0.1   |        0.798 |
| land      | PHYS + same-date offset [diagnostic only]   | 1000 |       1.508 |        0     |   1.508 |          2.51  |              1.59 |  0.1   |        0.798 |
| land      | PHYS, sigma_a(g)=0.25 % (analytic re-prior) | 1000 |       1.003 |       -0.782 |   1.272 |          1.248 |              1.59 |  0.602 |        0.772 |
| land      | E0_am (round 1/2)                           | 1000 |       2.014 |        1.852 |   2.736 |          2.813 |              1.59 | -0.605 |        0.675 |
| land      | E2cP_am_hgb (round 2, ML)                   | 1000 |       1.28  |        0.858 |   1.541 |          1.3   |              1.59 |  0.352 |        0.542 |
| land      | OB_E2cP_am_hgb (round 2, ML + blend)        | 1000 |       1.241 |        0.608 |   1.382 |          1.018 |              1.59 |  0.391 |        0.542 |

### Diagnostics

|                                 | 0       | 1       |
|:--------------------------------|:--------|:--------|
| surface                         | ocean   | land    |
| n                               | 1000    | 1000    |
| conv_rate                       | 1.0     | 1.0     |
| med_iter                        | 4.0     | 3.0     |
| med_nfwd                        | 35.0    | 31.0    |
| chi2_med                        | 8.76    | 10.732  |
| chi2_q90                        | 21.874  | 21.925  |
| chi2_per_obs_med                | 1.251   | 1.533   |
| dof_g_med                       | 0.927   | 0.917   |
| dof_g_q10                       | 0.877   | 0.878   |
| dof_g_q90                       | 0.935   | 0.935   |
| dof_total_med                   | 2.793   | 2.762   |
| sig_g_ppm_med                   | 1.078   | 1.152   |
| same_date_offset_ppm            | 2.582   | 3.802   |
| ms_med                          | 150.075 | 123.359 |
| ms_mean                         | 170.741 | 140.663 |
| tauL_med                        | 0.07    | 0.08    |
| sigmaH_med                      | 0.962   | 0.914   |
| alpha_med                       | 0.971   | 1.042   |
| h_med                           | 1.002   | 0.956   |
| s_p_med                         | 1.0     | 1.0     |
| g_med                           | 0.994   | 0.988   |
| frac_tauL_at_bound              | 0.0     | 0.0     |
| corr_tauL_aod_total             | 0.073   | 0.621   |
| spearman_tauL_aod               | -0.052  | 0.616   |
| spearman_sigmaH_dust_height     | -0.068  | 0.015   |
| spearman_h_h2o_scale            | 0.98    | 0.993   |
| corr_sp_psfc_ratio              | 0.128   | 0.286   |
| corr_g_minus1_vs_bc_minus_prior | 0.699   | 0.797   |

### Observables, ocean (medians)

|                |     obs |   F(prior state) |   resid (obs-F) med |   resid rsd |   pull rsd |   obs rsd |
|:---------------|--------:|-----------------:|--------------------:|------------:|-----------:|----------:|
| o2a_k1_am      |  0.9756 |           0.9587 |              0.0044 |      0.0057 |     1.3024 |    0.0098 |
| o2a_k2diag_am  | -0.0015 |           0.0111 |             -0.0082 |      0.0079 |     1.3247 |    0.005  |
| wco2_k1_am     |  0.9879 |           0.99   |              0.0005 |      0.002  |     0.8037 |    0.0029 |
| wco2_k1w_am    |  1.0059 |           0.9851 |              0.0097 |      0.0095 |     0.2925 |    0.0604 |
| sco2_k1_am     |  0.9992 |           1.003  |              0.0001 |      0.0034 |     1.1545 |    0.0061 |
| sco2_k1w_am    |  0.9997 |           0.991  |             -0.0053 |      0.0069 |     0.2273 |    0.0533 |
| sco2_k2diag_am |  0.0113 |           0.0122 |              0.0011 |      0.0038 |     1.185  |    0.0052 |

### Observables, land (medians)

|                |     obs |   F(prior state) |   resid (obs-F) med |   resid rsd |   pull rsd |   obs rsd |
|:---------------|--------:|-----------------:|--------------------:|------------:|-----------:|----------:|
| o2a_k1_am      |  0.9914 |           0.9855 |              0.0049 |      0.0039 |     0.9072 |    0.0108 |
| o2a_k2diag_am  | -0.0046 |           0.0068 |             -0.0112 |      0.0052 |     0.8217 |    0.0062 |
| wco2_k1_am     |  0.9837 |           0.9971 |             -0.0021 |      0.0019 |     0.6872 |    0.0103 |
| wco2_k1w_am    |  0.9758 |           0.9997 |              0.0173 |      0.0147 |     0.4082 |    0.1072 |
| sco2_k1_am     |  0.9937 |           1.0028 |              0.0007 |      0.0027 |     0.9152 |    0.0066 |
| sco2_k1w_am    |  0.946  |           1.0011 |             -0.0139 |      0.0088 |     0.2909 |    0.0924 |
| sco2_k2diag_am |  0.0057 |           0.0081 |             -0.002  |      0.0056 |     1.6589 |    0.0066 |

Reconstructed formal sigma / stored sig_* (median):

|                |   0 |
|:---------------|----:|
| sigchk_sco2_k1 |   1 |
| sigchk_o2a_k1  |   1 |
| sigchk_wco2_k1 |   1 |
