# Build Figure 6a and 6d country data from original R inputs and refreshed fits.
# Run after the fit engine with WCC_SCENARIO=ssp5.
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({library(readr); library(dplyr)})
base_dir <- normalizePath(Sys.getenv("WCC_BASE_DIR", unset = getwd()), winslash = "/", mustWork = TRUE)
sc <- "ssp5"
input_dir <- file.path(base_dir, "inputs"); out_dir <- file.path(base_dir, "r_outputs", sc)
fig_dir <- file.path(base_dir, "figure6_data"); dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
ctx <- read_csv(file.path(input_dir, "fit_input_ssp5_2050_v2.csv"), show_col_types = FALSE) %>% filter(year == 2050)
s7 <- read_csv(file.path(out_dir, "stage7_fit_country_summary.csv"), show_col_types = FALSE) %>% filter(year == 2050, scenario == sc)
s8 <- read_csv(file.path(out_dir, "stage8_bundle_country_summary.csv"), show_col_types = FALSE) %>% filter(year == 2050, scenario == sc)
map_data <- ctx %>%
  left_join(s7 %>% select(iso3, first_mode, first_mode_benchmark_pct, any_mode_good_hist75, any_mode_very_good_hist80), by = "iso3") %>%
  left_join(s8 %>% select(iso3, second_mode, third_mode, n_selected_modes, bundle_score, bundle_good_hist75), by = "iso3")
write_csv(map_data, file.path(fig_dir, "figure6a_ssp5_map_data.csv"))
cohort <- ctx %>%
  inner_join(s7, by = c("iso3", "country", "year", "scenario")) %>%
  left_join(s8 %>% select(iso3, second_mode, third_mode, n_selected_modes, bundle_score, bundle_good_hist75), by = "iso3")
risk_cut <- quantile(cohort$conflict_probability, .75, na.rm = TRUE)
precip_cut <- quantile(cohort$precipitation, .25, na.rm = TRUE)
bws_cut <- quantile(cohort$bws_score, .75, na.rm = TRUE)
gdp_cut <- quantile(cohort$gdp_constant_2015_us, .25, na.rm = TRUE)
cohort <- cohort %>% mutate(
  high_conflict_risk = conflict_probability >= risk_cut,
  low_precipitation = precipitation <= precip_cut,
  severe_water_stress = bws_score >= bws_cut,
  low_income = gdp_constant_2015_us <= gdp_cut,
  multi_mode_recommendation = n_selected_modes > 1
)
write_csv(cohort, file.path(fig_dir, "figure6d_ssp5_country_cohort_data.csv"))
set.seed(1234); B <- 1500
metrics <- c("any_mode_good_hist75", "any_mode_very_good_hist80", "multi_mode_recommendation")
labels <- c("At least one high-fit mode", "At least one very-high-fit mode", "More than one recommended mode")
groups <- list("All countries"=rep(TRUE,nrow(cohort)), "High conflict risk"=cohort$high_conflict_risk,
  "Low precipitation"=cohort$low_precipitation, "Severe water stress"=cohort$severe_water_stress, "Low income"=cohort$low_income)
out <- list(); k <- 1
for (g in names(groups)) {
  d <- cohort[groups[[g]], , drop=FALSE]
  for (j in seq_along(metrics)) {
    v <- as.logical(d[[metrics[j]]]); v <- v[!is.na(v)]; n <- length(v)
    est <- if(n) mean(v) else NA_real_
    boot <- if(n) replicate(B, mean(sample(v, size=n, replace=TRUE))) else numeric()
    ci <- if(n) quantile(boot, c(.025,.975), names=FALSE, na.rm=TRUE) else c(NA_real_,NA_real_)
    out[[k]] <- data.frame(group=g, metric=labels[j], n=n, estimate=est, ci_low=ci[1], ci_high=ci[2]); k <- k+1
  }
}
write_csv(bind_rows(out), file.path(fig_dir, "figure6d_ssp5_cohort_summary.csv"))
message("Wrote Figure 6a and 6d data to ", fig_dir)
