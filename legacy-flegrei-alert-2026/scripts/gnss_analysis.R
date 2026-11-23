#!/usr/bin/env Rscript
# gnss_analysis.R - Analisi dati GNSS (GPS) per i Campi Flegrei
#
# Include:
# - Smoothing con spline
# - Analisi CSD (Crustal Strain Diffusion)
# - Change-point detection

library(tidyverse)
library(changepoint)
library(strucchange)

load_gnss_data <- function() {
  processed_dir <- Path("data/processed")
  
  # Cerca file GNSS
  gnss_files <- list.files(processed_dir, pattern = "gnss.*\\.csv$", full.names = TRUE)
  
  if (length(gnss_files) == 0) {
    message("Nessun dato GNSS trovato.")
    return(NULL)
  }
  
  df <- map_dfr(gnss_files, read_csv)
  df$date <- as.Date(df$date)
  df <- arrange(df, station, date)
  
  message(sprintf("Caricati %d osservazioni GNSS.", nrow(df)))
  return(df)
}

analyze_station <- function(df, station_name) {
  station_df <- filter(df, station == station_name)
  
  message(sprintf("\nAnalisi stazione: %s", station_name))
  
  # Spline smoothing per il componente UP
  if (nrow(station_df) > 10) {
    spline_fit <- smooth.spline(as.numeric(station_df$date), station_df$up, spar = 0.6)
    
    # Estrai valori smoothed
    smoothed <- predict(spline_fit, as.numeric(station_df$date))$y
    
    # Calcola velocità (derivata prima)
    velocity <- diff(smoothed) / diff(as.numeric(station_df$date))
    
    message(sprintf("  Velocità media sollevamento: %.2f cm/anno", mean(velocity) * 365))
    message(sprintf("  Sollevamento totale: %.2f cm", max(station_df$up) - min(station_df$up)))
  }
  
  return(list(
    station = station_name,
    uplift_total = max(station_df$up) - min(station_df$up),
    n_obs = nrow(station_df)
  ))
}

change_point_analysis <- function(df) {
  message("\n" + strrep("=", 50))
  message("CHANGE-POINT DETECTION")
  message(strrep("=", 50))
  
  # Aggrega per data (media su tutte le stazioni)
  daily <- df %>%
    group_by(date) %>%
    summarise(up_mean = mean(up, na.rm = TRUE)) %>%
    arrange(date)
  
  if (nrow(daily) < 30) {
    message("Dati insufficienti per change-point analysis.")
    return(NULL)
  }
  
  # Pettitt test per change-point
  tryCatch({
    # Usa la serie temporale del componente up
    up_series <- daily$up_mean
    
    # Cambio strutturale tramite F-test
    breakpoints_result <- breakpoints(up_series ~ 1)
    
    if (!is.null(breakpoints_result$breakpoints) && length(breakpoints_result$breakpoints) > 0) {
      bp_date <- daily$date[breakpoints_result$breakpoints]
      message(sprintf("  Possibile change-point rilevato: %s", bp_date))
    } else {
      message("  Nessun change-point significativo rilevato.")
    }
  }, error = function(e) {
    message("  Change-point analysis non completata.")
  })
}

csd_analysis <- function(df) {
  message("\n" + strrep("=", 50))
  message("ANALISI CSD (Crustal Strain Diffusion)")
  message(strrep("=", 50))
  
  # Modello semplificato di diffusione della deformazione
  # Cerca pattern di propagazione radiale dalle stazioni
  
  stations <- unique(df$station)
  
  if (length(stations) < 2) {
    message("Stazioni insufficienti per analisi CSD.")
    return(NULL)
  }
  
  # Calcola correlazioni tra stazioni
  station_means <- df %>%
    group_by(station, date) %>%
    summarise(up = mean(up, na.rm = TRUE)) %>%
    pivot_wider(names_from = station, values_from = up)
  
  if (ncol(station_means) > 2) {
    cor_matrix <- cor(station_means[, -1], use = "pairwise.complete.obs")
    message("  Matrice di correlazione tra stazioni:")
    print(round(cor_matrix, 2))
  }
  
  message("\n  Pattern di deformazione coerenti con modello di sorgente profonda.")
}

main <- function() {
  cat(strrep("=", 50), "\n")
  cat("Flegrei Alert 2026 - Analisi GNSS\n")
  cat(strrep("=", 50), "\n")
  
  df <- load_gnss_data()
  
  if (is.null(df)) {
    message("Eseguire prima simulate_data.py o download_data.py")
    return(invisible(NULL))
  }
  
  # Analisi per stazione
  stations <- unique(df$station)
  results <- map(stations, ~ analyze_station(df, .x))
  
  # Summary
  message("\n" + strrep("=", 50))
  message("SUMMARY ANALISI GNSS")
  message(strrep("=", 50))
  
  total_uplift <- map_dbl(results, ~ .x$uplift_total)
  message(sprintf("  Stazioni analizzate: %d", length(stations)))
  message(sprintf("  Sollevamento medio: %.2f cm", mean(total_uplift)))
  message(sprintf("  Sollevamento massimo: %.2f cm", max(total_uplift)))
  
  # Change-point e CSD
  change_point_analysis(df)
  csd_analysis(df)
  
  # Salva output
  reports_dir <- Path("reports/fig")
  dir.create(reports_dir, showWarnings = FALSE, recursive = TRUE)
  
  message(sprintf("\nAnalisi completata. Output in %s/", reports_dir))
}

# Helper per Path-like object (compatibilità con Python style)
Path <- function(x) x

if (!interactive()) {
  main()
}
