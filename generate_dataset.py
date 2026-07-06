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
annual_rainfall        = np.random.uniform(500, 4000, n)
seasonal_rainfall      = annual_rainfall * np.random.uniform(0.4, 0.85, n)
monsoon_intensity      = np.random.uniform(0, 10, n)
cloud_visibility       = np.random.uniform(0.1, 10, n)
river_management       = np.random.uniform(0, 10, n)
deforestation          = np.random.uniform(0, 10, n)
urbanization           = np.random.uniform(0, 10, n)
climate_change         = np.random.uniform(0, 10, n)
drainage_systems       = np.random.uniform(0, 10, n)
coastal_vulnerability  = np.random.uniform(0, 10, n)
landslides             = np.random.uniform(0, 10, n)
watersheds             = np.random.uniform(0, 10, n)
infrastructure_decay   = np.random.uniform(0, 10, n)
population_score       = np.random.uniform(0, 10, n)
wetland_loss           = np.random.uniform(0, 10, n)
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
flood_prob = np.clip(flood_prob + np.random.normal(0, 0.03, n), 0, 1)
flood_label = (flood_prob >= 0.47).astype(int)
df = pd.DataFrame({
    "MonsoonIntensity":         monsoon_intensity.round(2),
    "TopographyDrainage":       (10 - drainage_systems).round(2),
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
    "FloodLabel":               flood_label
})
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/flood_prediction.csv", index=False)
print(f"Dataset created: dataset/flood_prediction.csv  ({len(df)} rows)")
print(f"Flood cases: {flood_label.sum()}  |  No-Flood cases: {(1-flood_label).sum()}")
