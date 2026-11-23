#!/usr/bin/env Rscript
# geo_chemistry.R - Analisi dati geochimici per i Campi Flegrei
#
# Include:
# - Analisi cointegrazione (CO2/SO2, temperatura)
# - Test di causalità di Granger
# - Correlazioni tra parametri

library(tidyverse)
library(vars)
library(tseries)

load_gas_data <- function() {
  processed_dir <- Path("data/processed")
  
  # Cerca file geochimici
  gas_files <- list.files(processed_dir, pattern = "gas.*\\.csv$", full.names = TRUE)
  
  if (length(gas_files) == 0) {
    message("Nessun dato geochimico trovato.")
    return(NULL)
  }
  
  df <- map_dfr(gas_files, read_csv)
  df$date <- as.Date(df$date)
  df <- arrange(df, station, date)
  
  message(sprintf("Caricati %d osservazioni geochimiche.", nrow(df)))
  return(df)
}

station_analysis <- function(df, station_name) {
  station_df <- filter(df, station == station_name)
  
  message(sprintf("\nAnalisi stazione: %s", station_name))
  
  # Statistiche descrittive
  message("  Statistiche CO2/SO2 ratio:")
  message(sprintf("    Media: %.2f", mean(station_df$co2_so2_ratio, na.rm = TRUE)))
  message(sprintf("    Std Dev: %.2f", sd(station_df$co2_so2_ratio, na.rm = TRUE)))
  message(sprintf("    Min: %.2f", min(station_df$co2_so2_ratio, na.rm = TRUE)))
  message(sprintf("    Max: %.2f", max(station_df$co2_so2_ratio, na.rm = TRUE)))
  
  # Trend temperatura
  if (nrow(station_df) > 30) {
    temp_model <- lm(temperature ~ as.numeric(date), data = station_df)
    message(sprintf("  Trend temperatura: %.3f °C/giorno", coef(temp_model)[2]))
  }
  
  return(list(
    station = station_name,
    co2_so2_mean = mean(station_df$co2_so2_ratio, na.rm = TRUE),
    temp_trend = if(nrow(station_df) > 30) coef(lm(temperature ~ as.numeric(date), data = station_df))[2] else NA
  ))
}

granger_causality_test <- function(df) {
  message("\n" + strrep("=", 50))
  message("TEST DI CAUSALITÀ DI GRANGER")
  message(strrep("=", 50))
  
  # Aggrega per data
  daily <- df %>%
    group_by(date) %>%
    summarise(
      co2_so2 = mean(co2_so2_ratio, na.rm = TRUE),
      temperature = mean(temperature, na.rm = TRUE),
      pressure = mean(pressure, na.rm = TRUE)
    ) %>%
    arrange(date)
  
  if (nrow(daily) < 50) {
    message("Dati insufficienti per test di Granger (minimo 50 osservazioni).")
    return(NULL)
  }
  
  # Crea time series object
  ts_data <- ts(daily[, c("co2_so2", "temperature")], frequency = 7)
  
  # Test di stazionarietà (ADF)
  message("\nTest di stazionarietà (ADF):")
  adf_co2 <- tryCatch({
    adf.test(ts_data[, 1])
  }, error = function(e) NULL)
  
  if (!is.null(adf_co2)) {
    message(sprintf("  CO2/SO2: p-value = %.4f %s", 
                    adf_co2$p.value,
                    if(adf_co2$p.value < 0.05) "(stazionario)" else "(non stazionario)"))
  }
  
  # VAR model e Granger test
  tryCatch({
    # Selezione lag ottimale
    var_select <- VARselect(ts_data, lag.max = 10, type = "const")
    best_lag <- var_select$selection[["AIC(n)"]]
    
    message(sprintf("\nLag ottimale (AIC): %d", best_lag))
    
    # Fit VAR
    var_model <- VAR(ts_data, p = best_lag, type = "const")
    
    # Test di Granger
    granger_result <- causality(var_model, cause = "temperature")
    
    if (!is.null(granger_result$Granger$Chi2)) {
      p_value <- granger_result$Granger$Chisq[4]  # p-value del test
      message(sprintf("\nGranger causality (Temp -> CO2/SO2): p-value = %.4f", p_value))
      
      if (p_value < 0.05) {
        message("  ✓ La temperatura causa (Granger) variazioni nel rapporto CO2/SO2")
      } else {
        message("  ✗ Nessuna evidenza di causalità di Granger")
      }
    }
  }, error = function(e) {
    message("  Test di Granger non completato.")
  })
}

cointegration_analysis <- function(df) {
  message("\n" + strrep("=", 50))
  message("ANALISI DI COINTEGRAZIONE")
  message(strrep("=", 50))
  
  # Aggrega per data
  daily <- df %>%
    group_by(date) %>%
    summarise(
      co2_so2 = mean(co2_so2_ratio, na.rm = TRUE),
      temperature = mean(temperature, na.rm = TRUE)
    ) %>%
    arrange(date) %>%
    na.omit()
  
  if (nrow(daily) < 50) {
    message("Dati insufficienti per analisi di cointegrazione.")
    return(NULL)
  }
  
  # Test di cointegrazione di Engle-Granger (semplificato)
  message("\nRegressione CO2/SO2 ~ Temperatura:")
  
  model <- lm(co2_so2 ~ temperature, data = daily)
  summary_model <- summary(model)
  
  message(sprintf("  Coefficiente temperatura: %.4f", coef(model)[2]))
  message(sprintf("  R²: %.3f", summary_model$r.squared))
  
  # Test sui residui (ADF)
  residuals_ts <- ts(residuals(model))
  
  tryCatch({
    adf_resid <- adf.test(residuals_ts)
    message(sprintf("\nADF test sui residui: p-value = %.4f", adf_resid$p.value))
    
    if (adf_resid$p.value < 0.05) {
      message("  ✓ I residui sono stazionari: le serie sono cointegrate")
    } else {
      message("  ✗ I residui non sono stazionari: nessuna cointegrazione")
    }
  }, error = function(e) {
    message("  Test ADF sui residui non completato.")
  })
}

main <- function() {
  cat(strrep("=", 50), "\n")
  cat("Flegrei Alert 2026 - Analisi Geochimica\n")
  cat(strrep("=", 50), "\n")
  
  df <- load_gas_data()
  
  if (is.null(df)) {
    message("Eseguire prima simulate_data.py o download_data.py")
    return(invisible(NULL))
  }
  
  # Analisi per stazione
  stations <- unique(df$station)
  results <- map(stations, ~ station_analysis(df, .x))
  
  # Summary
  message("\n" + strrep("=", 50))
  message("SUMMARY ANALISI GEOCHIMICA")
  message(strrep("=", 50))
  
  co2_means <- map_dbl(results, ~ .x$co2_so2_mean)
  message(sprintf("  Stazioni analizzate: %d", length(stations)))
  message(sprintf("  Rapporto CO2/SO2 medio: %.2f", mean(co2_means)))
  
  # Test avanzati
  granger_causality_test(df)
  cointegration_analysis(df)
  
  # Salva output
  reports_dir <- Path("reports/fig")
  dir.create(reports_dir, showWarnings = FALSE, recursive = TRUE)
  
  message(sprintf("\nAnalisi completata. Output in %s/", reports_dir))
}

# Helper per Path-like object
Path <- function(x) x

if (!interactive()) {
  main()
}
