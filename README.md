# Cross-site HAWT energy estimation with Extra Trees (Cajicá → João Pessoa)

Code, data, trained models and outputs of the paper *"Cross-Site Prospective Estimation of HAWT Energy Output Using a Transferable Extra Trees Model: From High-Altitude Colombia to Coastal Brazil"* (IREA, manuscript ID 28006, revised version).

An Extra Trees regressor is trained on one-minute meteorological data and RMS voltage of a Pikasola 400 W horizontal-axis wind turbine measured at Cajicá, Colombia (2,558 m a.s.l., 17 June – 19 August 2022), validated with chronologically blocked cross-validation, and applied without retraining to ten-minute data of the GITUD_UFPB weather station in João Pessoa, Brazil (April–May 2025). **The João Pessoa results are unvalidated exploratory projections: there are no turbine measurements at that site.**

## Contents

| Path | What it is |
|---|---|
| `modelo_eolico_paso_a_paso.ipynb` | Step-by-step tutorial: data loading, cleaning, model training, export of dataset and model, projection for João Pessoa |
| `02_replica_validacion_cronologica.ipynb` | Replica of the original pipeline (random split) and the chronological validation of the 20 models |
| `03_modelo_transferible_jpa.ipynb` | Final model with inputs available at both sites, out-of-distribution diagnostics, projection, Tables II and V |
| `scripts/01…04_*.py` | Figures and complementary analyses of the paper (run in this order from the repository root) |
| `data/data_cajica_2022_crudo.csv` | Raw Cajicá record (turbine voltage and weather) |
| `data/estacion_jpa_2025_crudo.csv` | Raw export of the João Pessoa station (tab-separated, no header) |
| `data/dataset_cajica_limpio.csv` | Cleaned Cajicá training table (real observations) |
| `models/et_tablaIV_vrms.joblib` (+ `_meta.json`) | Final Extra Trees model (8 inputs of the paper) |
| `models/extra_trees_vrms.joblib` (+ `_meta.json`) | Model with the 8 variables of the original notebook (needs soil moisture and a vertical-axis variable not available in João Pessoa) |
| `outputs/` | Tables (CSV/LaTeX), predictions, summaries; `outputs/figuras_articulo/` holds the figures of the paper |

## Method in short

* Cleaning: physical limits set to zero, relative humidity outside 30–70 % replaced by the mean, duplicated timestamps removed, one-minute grid with backward fill (only real observations are scored).
* Inputs (equivalent at both sites): wind angle, `velU = mag·cos θ`, `velV = mag·sin θ`, air density from temperature and altitude, relative humidity, hour of day, `xmag_disc` (wind speed if ≥ 3.5 m/s, else 0) and `TEMP_norm` (temperature if ≥ 23 °C, else 0).
* Validation: expanding-window `TimeSeriesSplit(n_splits=10, gap=1440)`; Extra Trees (100 trees, `random_state=123`): R² = 0.44, MAE = 0.38 V; random 10-fold gives R² = 0.76 (optimistic).
* João Pessoa: wind speed converted from km/h (`/3.6`), wind direction is a compass bearing in degrees; energy `E = V²·Δt/R` with `R = 100 Ω`.

## Data provenance

* Cajicá: experimental campaign of the Universidad Distrital Francisco José de Caldas.
* João Pessoa: Meteoagro MA 3500 automatic weather station (https://www.meteoagro.co/products/detail/23/), station GITUD_UFPB (UFPB).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute 03_modelo_transferible_jpa.ipynb
python scripts/01_figuras_cajica_validacion.py
python scripts/02_figuras_cajica_comparacion_energia.py
python scripts/03_figuras_joao_pessoa.py
python scripts/04_analisis_complementario.py
```

XGBoost and LightGBM need the OpenMP runtime (`libomp`) on macOS.

## Licence and citation

No licence has been chosen yet; until one is added, all rights are reserved by the authors. Please cite the paper if you use these materials.
