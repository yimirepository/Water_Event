library(dplyr)
library(ggplot2)
library(readr)

data_dir <- Sys.getenv("FIGURE_DATA_DIR", unset = "data")
cohort_csv <- file.path(data_dir, "figure10", "figure10_cohort_data.csv")
output_dir <- file.path("outputs", "figure10")

SEED <- 1234L
N_BOOT <- 1000L

# Sort by ISO3 before bootstrapping so the random draws are reproducible.
countries <- read_csv(cohort_csv, show_col_types = FALSE) |>
  select(
    iso3, high_conflict_risk, severe_water_stress, low_income,
    any_mode_good_hist75, any_mode_very_good_hist80, n_selected_modes
  ) |>
  arrange(iso3) |>
  mutate(multi_mode = n_selected_modes > 1L)
stopifnot(nrow(countries) == 152L, !anyDuplicated(countries$iso3), !anyNA(countries))

country_groups <- list(
  "All countries" = countries,
  "High-conflict-risk countries" = filter(countries, high_conflict_risk),
  "Severe-water-stress countries" = filter(countries, severe_water_stress),
  "Low-income countries" = filter(countries, low_income)
)
metrics <- c(
  "Highly fitted measures identified" = "any_mode_good_hist75",
  "Very highly fitted measures identified" = "any_mode_very_good_hist80",
  "Combination of >1 modes recommended" = "multi_mode"
)
stopifnot(all(vapply(country_groups, nrow, integer(1)) > 0L))

# Use the original percentile bootstrap over countries for each group and metric.
bootstrap_proportion <- function(values) {
  set.seed(SEED)
  n <- length(values)
  bootstrap <- replicate(N_BOOT, {
    idx <- sample.int(n, size = n, replace = TRUE)
    mean(values[idx])
  })
  interval <- quantile(bootstrap, c(0.025, 0.975), names = FALSE)
  tibble(estimate = mean(values), ci_low = interval[1], ci_high = interval[2])
}

summary_data <- bind_rows(lapply(names(country_groups), function(group) {
  group_data <- country_groups[[group]]
  bind_rows(lapply(names(metrics), function(metric) {
    bootstrap_proportion(group_data[[metrics[[metric]]]]) |>
      mutate(country_group = group, metric_label = metric)
  }))
})) |>
  mutate(
    country_group = factor(country_group, levels = names(country_groups)),
    metric_label = factor(metric_label, levels = names(metrics))
  )

draw_panel <- function(data) {
  ggplot(data, aes(x = metric_label, y = estimate, fill = country_group)) +
    geom_col(
      position = position_dodge(width = 1.1), width = 0.50, colour = "white"
    ) +
    geom_errorbar(
      aes(ymin = ci_low, ymax = ci_high),
      position = position_dodge(width = 1.1), width = 0.25, linewidth = 0.70
    ) +
    coord_flip() +
    scale_fill_manual(values = c(
      "All countries" = "#8C3F74",
      "High-conflict-risk countries" = "#007A8A",
      "Severe-water-stress countries" = "#C75B12",
      "Low-income countries" = "#1F4E79"
    )) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      limits = c(0, 1), expand = expansion(mult = c(0, 0.02))
    ) +
    theme_minimal(base_family = "Arial") +
    theme(
      text = element_text(family = "Arial", size = 28),
      legend.position = "none",
      axis.text.x = element_text(size = 18),
      axis.text.y = element_blank(),
      axis.title = element_text(size = 22),
      axis.ticks.x = element_line(colour = "black"),
      axis.ticks.y = element_blank(),
      axis.ticks.length = grid::unit(3, "mm"),
      panel.grid = element_blank(),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 1.5)
    )
}

output_files <- c(
  "figure10_1_high_fit.png",
  "figure10_2_very_high_fit.png",
  "figure10_3_multi_modes.png"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
for (i in seq_along(metrics)) {
  panel <- draw_panel(filter(summary_data, metric_label == names(metrics)[i]))
  ggsave(
    filename = file.path(output_dir, output_files[i]),
    plot = panel, width = 6, height = 6, dpi = 300, device = ragg::agg_png
  )
}
message("Saved three Figure 10 panels to: ", output_dir)

combined <- ggplot(summary_data, aes(x = country_group, y = estimate, fill = country_group)) +
  geom_col(width = 0.45) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.18) +
  coord_flip() +
  facet_grid(cols = vars(metric_label), switch = "x", labeller = label_wrap_gen(24)) +
  scale_fill_manual(values = c("#8C3F74", "#007A8A", "#C75B12", "#1F4E79")) +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1),
                     breaks = seq(0, 1, 0.25)) +
  labs(x = NULL, y = "Estimate") +
  theme_bw() +
  theme(legend.position = "none", panel.grid = element_blank(),
        strip.placement = "outside", strip.background = element_blank())
ggsave(file.path(output_dir, "figure10.png"), combined,
       width = 12, height = 3.8, dpi = 300, device = ragg::agg_png)
write_csv(summary_data, file.path(output_dir, "figure10_cohort_estimates.csv"))
