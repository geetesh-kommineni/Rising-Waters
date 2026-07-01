"""
generate_dataset.py
Generates a realistic synthetic flood prediction dataset for the Rising Waters project.
Run this once to create dataset/flood_prediction.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 2500

# --- Feature generation (realistic meteorological/environmental ranges) ---
annual_rainfall        = np.random.uniform(500, 4000, n)       # mm
seasonal_rainfall      = annual_rainfall * np.random.uniform(0.4, 0.85, n)
monsoon_intensity      = np.random.uniform(0, 10, n)           # 0-10 index
cloud_visibility       = np.random.uniform(0.1, 10, n)         # km
river_management       = np.random.uniform(0, 10, n)           # quality index 0-10
deforestation          = np.random.uniform(0, 10, n)           # severity 0-10
urbanization           = np.random.uniform(0, 10, n)           # 0-10 index
climate_change         = np.random.uniform(0, 10, n)           # impact 0-10
drainage_systems       = np.random.uniform(0, 10, n)           # quality 0-10
coastal_vulnerability  = np.random.uniform(0, 10, n)           # risk 0-10
landslides             = np.random.uniform(0, 10, n)           # risk 0-10
watersheds             = np.random.uniform(0, 10, n)           # health 0-10
infrastructure_decay   = np.random.uniform(0, 10, n)           # deterioration 0-10
population_score       = np.random.uniform(0, 10, n)           # density risk 0-10
wetland_loss           = np.random.uniform(0, 10, n)           # % loss index 0-10

# --- Target: flood probability (continuous 0–1) based on realistic drivers ---
flood_prob = (
    0.25 * (annual_rainfall / 4000) +
    0.20 * (seasonal_rainfall / 3400) +
    0.10 * (monsoon_intensity / 10) +
    0.08 * (1 - cloud_visibility / 10) +
    0.07 * (1 - river_management / 10) +
    0.06 * (deforestation / 10) +
    0.05 * (urbanization / 10) +
    0.04 * (climate_change / 10) +
    0.04 * (1 - drainage_systems / 10) +
    0.04 * (coastal_vulnerability / 10) +
    0.03 * (landslides / 10) +
    0.02 * (1 - watersheds / 10) +
    0.01 * (infrastructure_decay / 10) +
    0.01 * (population_score / 10) +
    0.00 * (wetland_loss / 10)
)
# Clip and add slight noise
flood_prob = np.clip(flood_prob + np.random.normal(0, 0.03, n), 0, 1)

# Binary label: 1 = Flood, 0 = No Flood  (threshold ~0.47 for ~50/50 split)
flood_label = (flood_prob >= 0.47).astype(int)

df = pd.DataFrame({
    "MonsoonIntensity":         monsoon_intensity.round(2),
    "TopographyDrainage":       (10 - drainage_systems).round(2),   # higher = worse
    "RiverManagement":          river_management.round(2),
    "Deforestation":            deforestation.round(2),
    "Urbanization":             urbanization.round(2),
    "ClimateChange":            climate_change.round(2),
    "DamQuality":               (10 - infrastructure_decay).round(2),
    "Siltation":                np.random.uniform(0, 10, n).round(2),
    "AgriculturalPractices":    np.random.uniform(0, 10, n).round(2),
    "Encroachments":            np.random.uniform(0, 10, n).round(2),
    "IneffectiveDisasterPrep":  np.random.uniform(0, 10, n).round(2),
    "DrainageSystems":          drainage_systems.round(2),
    "CoastalVulnerability":     coastal_vulnerability.round(2),
    "Landslides":               landslides.round(2),
    "Watersheds":               watersheds.round(2),
    "DeterioratingInfrastructure": infrastructure_decay.round(2),
    "PopulationScore":          population_score.round(2),
    "WetlandLoss":              wetland_loss.round(2),
    "InadequatePlanning":       np.random.uniform(0, 10, n).round(2),
    "PoliticalFactors":         np.random.uniform(0, 10, n).round(2),
    "AnnualRainfall":           annual_rainfall.round(2),
    "SeasonalRainfall":         seasonal_rainfall.round(2),
    "CloudVisibility":          cloud_visibility.round(2),
    "FloodProbability":         flood_prob.round(4),
    "FloodLabel":               flood_label        # Target
})

os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/flood_prediction.csv", index=False)
print(f"Dataset created: dataset/flood_prediction.csv  ({len(df)} rows)")
print(f"Flood cases: {flood_label.sum()}  |  No-Flood cases: {(1-flood_label).sum()}")
