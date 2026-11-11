library(readxl)
library(sf)

project_dir <- "E:/AExperiment/11Water"
figure_data_dir <- file.path(
  project_dir, "F_Figure_New/score_refresh_package/figure6_data"
)
old_shp <- file.path(
  project_dir, "F_Picture/Figure6/20260422_Figure6_new/fig6a/fig6a/mode_points.shp"
)
world_shp <- "E:/Database/_CorrectWorldMap/WorldMap.shp"

output_dir <- "E:/AExperiment/11Water/F_Figure_New/Figure8"
output_shp <- file.path(output_dir, "mode_points_updated.shp")

# Read the country list and expected recommendation order from Figure 6c.
countries <- read_excel(
  file.path(figure_data_dir, "figure5_and_figure6c_tables.xlsx"),
  sheet = "Fig 6c table",
  range = "A5:E28"
)
countries <- countries[order(countries$Order), ]
stopifnot(nrow(countries) == 23L, !anyDuplicated(countries$ISO3))

scores <- read.csv(
  file.path(figure_data_dir, "figure6a_ssp5_map_data.csv"),
  na.strings = c("", "NA"),
  stringsAsFactors = FALSE
)
scores <- subset(scores, year == 2050 & scenario == "ssp5")
stopifnot(!anyDuplicated(scores$iso3))
score_rows <- match(countries$ISO3, scores$iso3)
stopifnot(!anyNA(score_rows))
scores <- scores[score_rows, ]
stopifnot(identical(as.character(countries$Country), scores$country))

mode_labels <- c(
  cross_border_basin_agreements = "Basin pact",
  joint_water_allocation_models = "Water allocation",
  transboundary_water_quality_standards = "Quality standard"
)
mode_full_names <- c(
  cross_border_basin_agreements = "Cross-Border Basin Agreements",
  joint_water_allocation_models = "Joint Water Allocation Models",
  transboundary_water_quality_standards = "Transboundary Water Quality Standards"
)

# Use the original centroids wherever possible.
old_points <- st_read(old_shp, quiet = TRUE)
centroids <- unique(st_drop_geometry(old_points)[, c("iso3", "X", "Y")])
stopifnot(!anyDuplicated(centroids$iso3))
new_codes <- setdiff(countries$ISO3, centroids$iso3)

# Compute centroids for the five added countries on the ArcGIS basemap.
# The France feature here represents metropolitan France.
world <- st_read(world_shp, quiet = TRUE)
world <- world[
  world$ISO %in% new_codes & (world$ISO != "FRA" | world$country2 == "France"),
]
stopifnot(setequal(world$ISO, new_codes), !anyDuplicated(world$ISO))
new_centroids <- st_geometry(world) |>
  st_transform(3857) |>
  st_centroid() |>
  st_transform(4326) |>
  st_coordinates()
centroids <- rbind(
  centroids,
  data.frame(iso3 = world$ISO, X = new_centroids[, "X"], Y = new_centroids[, "Y"])
)

# Use the original offsets for countries with multiple recommendations.
offsets <- list(
  data.frame(dx = 0, dy = 0),
  data.frame(dx = c(-1.8, 1.8), dy = c(0, 0)),
  data.frame(dx = c(-2.2, 0, 2.2), dy = c(-0.7, 1.3, -0.7))
)
mode_columns <- c("first_mode", "second_mode", "third_mode")
point_rows <- lapply(seq_len(nrow(countries)), function(i) {
  code <- countries$ISO3[i]
  mode_ids <- unlist(scores[i, mode_columns], use.names = FALSE)
  mode_ids <- mode_ids[!is.na(mode_ids)]
  n <- length(mode_ids)
  expected <- trimws(strsplit(countries[["Recommended modes (in order)"]][i], ";")[[1]])
  stopifnot(
    n > 0L,
    n == countries[["Number of recommended modes"]][i],
    n == scores$n_selected_modes[i],
    identical(unname(mode_full_names[mode_ids]), expected),
    !is.na(scores$conflict_probability[i])
  )

  center <- centroids[match(code, centroids$iso3), ]
  data.frame(
    iso3 = code,
    country = countries$Country[i],
    risk_value = scores$conflict_probability[i],
    mode_rank = paste0("mode_", seq_len(n)),
    mode_short = unname(mode_labels[mode_ids]),
    X = center$X,
    Y = center$Y,
    n_modes = n,
    idx = seq_len(n),
    lon_plot = center$X + offsets[[n]]$dx,
    lat_plot = center$Y + offsets[[n]]$dy
  )
})
points <- do.call(rbind, point_rows)

# Keep the manually adjusted US map position from the original point layer.
old_usa <- st_coordinates(old_points[
  old_points$iso3 == "USA" & old_points$mode_rank == "mode_1",
])
usa <- points$iso3 == "USA" & points$mode_rank == "mode_1"
stopifnot(nrow(old_usa) == 1L, sum(usa) == 1L)
points$lon_plot[usa] <- old_usa[1, "X"]
points$lat_plot[usa] <- old_usa[1, "Y"]

points <- st_as_sf(points, coords = c("lon_plot", "lat_plot"), crs = 4326)
stopifnot(
  nrow(points) == 41L,
  sum(points$mode_rank == "mode_1") == 23L,
  sum(points$mode_rank == "mode_2") == 18L,
  all(st_is_valid(points))
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
st_write(points, output_shp, layer_options = "ENCODING=UTF-8", quiet = TRUE)
message("Saved ", nrow(points), " points for ", nrow(countries), " countries: ", output_shp)
