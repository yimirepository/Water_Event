library(sf)

data_dir <- Sys.getenv("FIGURE_DATA_DIR", unset = "data")
input_file <- file.path(data_dir, "figure08", "figure08_map_data.csv")
output_dir <- file.path("outputs", "figure08")
output_shp <- file.path(output_dir, "mode_points_updated.shp")

countries <- read.csv(input_file, na.strings = c("", "NA"))
stopifnot(
  nrow(countries) == 23L,
  !anyDuplicated(countries$iso3),
  !anyNA(countries[c("iso3", "country", "risk_value", "first_mode", "X", "Y")]),
  identical(is.na(countries$first_lon_override), is.na(countries$first_lat_override))
)

mode_labels <- c(
  cross_border_basin_agreements = "Basin pact",
  joint_water_allocation_models = "Water allocation",
  transboundary_water_quality_standards = "Quality standard"
)

# Preserve the original offsets for countries with two recommendations.
point_rows <- lapply(seq_len(nrow(countries)), function(i) {
  country <- countries[i, ]
  mode_ids <- c(country$first_mode, country$second_mode)
  mode_ids <- mode_ids[!is.na(mode_ids)]
  stopifnot(all(mode_ids %in% names(mode_labels)))
  n <- length(mode_ids)
  lon <- country$X + if (n == 2L) c(-1.8, 1.8) else 0
  lat <- rep(country$Y, n)
  if (!is.na(country$first_lon_override)) {
    lon[1] <- country$first_lon_override
    lat[1] <- country$first_lat_override
  }
  data.frame(
    iso3 = country$iso3,
    country = country$country,
    risk_value = country$risk_value,
    mode_rank = paste0("mode_", seq_len(n)),
    mode_short = unname(mode_labels[mode_ids]),
    X = country$X,
    Y = country$Y,
    n_modes = n,
    idx = seq_len(n),
    lon_plot = lon,
    lat_plot = lat
  )
})

points <- st_as_sf(do.call(rbind, point_rows),
                   coords = c("lon_plot", "lat_plot"), crs = 4326)
stopifnot(nrow(points) == 41L, sum(points$mode_rank == "mode_2") == 18L,
          all(st_is_valid(points)))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
st_write(points, output_shp, layer_options = "ENCODING=UTF-8",
         delete_layer = TRUE, quiet = TRUE)
message("Saved ", nrow(points), " points for ", nrow(countries), " countries.")
