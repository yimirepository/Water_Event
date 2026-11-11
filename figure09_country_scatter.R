library(dplyr)
library(ggplot2)

data_dir <- Sys.getenv("FIGURE_DATA_DIR", unset = "data")
figure9_dir <- file.path("outputs", "figure09")
scatter_csv <- file.path(data_dir, "figure09", "figure09_scatter_data.csv")

# An optional first argument can redirect the two exports.
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0L) args[1] else figure9_dir

# The table already contains the final 2050 SSP5 cohort and the country labels.
plot_data <- read.csv(scatter_csv, na.strings = c("", "NA"))
stopifnot(
  nrow(plot_data) == 152L,
  !anyNA(plot_data),
  !anyDuplicated(plot_data$iso3),
  sum(plot_data$highlighted) == 23L
)

winsorize <- function(values) {
  cutoffs <- quantile(values, c(0.02, 0.98))
  pmin(pmax(values, cutoffs[1]), cutoffs[2])
}

# The main panel uses the original winsorization rule.
main_data <- plot_data |>
  mutate(x = winsorize(precipitation), y = winsorize(gdp_per_capita))

# The zoom panel uses untransformed values and the original window.
zoom_data <- plot_data |>
  filter(precipitation >= 0, precipitation < 2000,
         gdp_per_capita >= 0, gdp_per_capita < 50000) |>
  mutate(x = precipitation, y = gdp_per_capita)

# Expand the main x-axis so all selected countries can be shown.
main_x_max <- max(8000, ceiling(max(main_data$x[main_data$highlighted]) / 1000) * 1000)
axis_label <- scales::label_number(big.mark = ",")

draw_scatter <- function(data, background_alpha, highlight_alpha, label_size,
                         box_padding, point_padding, label_codes) {
  ggplot(data, aes(x = x, y = y)) +
    geom_point(
      data = filter(data, !highlighted),
      aes(size = risk, colour = risk), alpha = background_alpha
    ) +
    geom_point(
      data = filter(data, highlighted),
      aes(size = risk, colour = risk), alpha = highlight_alpha
    ) +
    
    ggrepel::geom_text_repel(
      data = filter(data, iso3 %in% label_codes),
      aes(label = country), size = label_size, seed = 1234,
      box.padding = box_padding, point.padding = point_padding,
      max.overlaps = Inf
    ) +
    scale_colour_gradientn(colours = c("#AFC6DD", "#EDC2A8", "#C75B12", "#8C3F74")) +
    scale_size_continuous(range = c(1.8, 6.2)) +
    labs(
      colour = "Conflict probability", size = "Conflict probability",
      x = "Projected precipitation (mm)",
      y = "Projected GDP per capita (constant 2015 US$)"
    ) +
    theme_minimal(base_size = 12.5) +
    theme(
      axis.title = element_text(size = 11.5, face = "bold", colour = "#444444"),
      axis.text = element_text(size = 10.5, colour = "#444444"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#D9D9D9", linewidth = 0.25),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
      legend.position = "none"
    )
}

zoom_codes <- zoom_data$iso3[zoom_data$highlighted]
main_codes <- setdiff(plot_data$iso3[plot_data$highlighted], zoom_codes)
p_main <- draw_scatter(main_data, 0.20, 0.99, 3.4, 0.24, 0.16, main_codes) +
  annotate("rect", xmin = 0, xmax = 2000, ymin = 0, ymax = 50000,
           fill = NA, linetype = "dashed") +
  scale_x_continuous(labels = axis_label, limits = c(NA, main_x_max)) +
  scale_y_continuous(labels = axis_label) +
  theme(legend.position = "top")

p_zoom <- draw_scatter(zoom_data, 0.25, 0.95, 3.2, 0.20, 0.14, zoom_codes) +
  scale_x_continuous(labels = axis_label, limits = c(0, 2000), expand = c(0.05, 0)) +
  scale_y_continuous(labels = axis_label, limits = c(0, 51000), expand = c(0, 0.02))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(
  file.path(output_dir, "figure9_main_23countries.png"), p_main,
  width = 9, height = 5.8, dpi = 600, device = ragg::agg_png
)
ggsave(
  file.path(output_dir, "figure9_zoom_23countries.png"), p_zoom,
  width = 7, height = 5.8, dpi = 600, device = ragg::agg_png
)
message(
  "Main panel: ", sum(main_data$highlighted),
  " highlighted countries. Zoom panel: ", sum(zoom_data$highlighted), "."
)

combined <- patchwork::wrap_plots(p_zoom, p_main, widths = c(1, 1.3))
ggsave(file.path(output_dir, "figure09.png"), combined,
       width = 13, height = 5.8, dpi = 600, device = ragg::agg_png)
