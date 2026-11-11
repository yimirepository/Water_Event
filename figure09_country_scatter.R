library(dplyr)
library(ggplot2)
library(readxl)

figures_dir <- "E:/AExperiment/11Water/F_Figure_New"
figure9_dir <- file.path(figures_dir, "Figure9")
figure_data_dir <- file.path(figures_dir, "score_refresh_package", "figure6_data")
country_workbook <- file.path(figure_data_dir, "figure5_and_figure6c_tables.xlsx")
cohort_csv <- file.path(figure_data_dir, "figure6d_ssp5_country_cohort_data.csv")

# An optional first argument can redirect the two exports.
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0L) args[1] else figure9_dir

# Read the country names and ISO3 codes from the Figure 6c table.
selected_countries <- read_excel(
  country_workbook, sheet = "Fig 6c table", range = "A5:C28"
) |>
  transmute(iso3 = trimws(ISO3), country = trimws(Country))
stopifnot(
  nrow(selected_countries) == 23L,
  !anyNA(selected_countries),
  all(nzchar(selected_countries$iso3) & nzchar(selected_countries$country)),
  !anyDuplicated(selected_countries$iso3)
)

cohort <- read.csv(cohort_csv, na.strings = c("", "NA"))
plot_data <- cohort |>
  filter(year == 2050, scenario == "ssp5") |>
  transmute(
    iso3, country,
    precipitation,
    gdp_per_capita = gdp_constant_2015_us,
    risk = conflict_probability,
    highlighted = iso3 %in% selected_countries$iso3
  ) |>
  filter(complete.cases(iso3, country, precipitation, gdp_per_capita, risk))
stopifnot(
  !anyDuplicated(plot_data$iso3),
  setequal(plot_data$iso3[plot_data$highlighted], selected_countries$iso3)
)

# Use the workbook spelling of highlighted country names.
highlight_rows <- match(selected_countries$iso3, plot_data$iso3)
plot_data$country[highlight_rows] <- selected_countries$country

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
                         box_padding, point_padding) {
  ggplot(data, aes(x = x, y = y)) +
    geom_point(
      data = filter(data, !highlighted),
      aes(size = risk, colour = risk), alpha = background_alpha
    ) +
    geom_point(
      data = filter(data, highlighted),
      aes(size = risk, colour = risk), alpha = highlight_alpha
    ) +
    
    #ggrepel::geom_text_repel(
    #  data = filter(data, highlighted),
    #  aes(label = country),
    #  size = label_size, fontface = "bold", colour = "black",
    #  box.padding = box_padding, point.padding = point_padding,
    #  segment.color = "#444444", max.overlaps = Inf, seed = 1234
    #) +
    
    scale_colour_gradientn(colours = c("#AFC6DD", "#EDC2A8", "#C75B12", "#8C3F74")) +
    scale_size_continuous(range = c(1.8, 6.2)) +
    labs(
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

p_main <- draw_scatter(main_data, 0.20, 0.99, 3.4, 0.24, 0.16) +
  scale_x_continuous(labels = axis_label, limits = c(NA, main_x_max)) +
  scale_y_continuous(labels = axis_label)

p_zoom <- draw_scatter(zoom_data, 0.25, 0.95, 3.2, 0.20, 0.14) +
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
