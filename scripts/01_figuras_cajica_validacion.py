import sys, math, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
warnings.filterwarnings("ignore")
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor

P = Path(__file__).resolve().parents[1]; OUT = P/"outputs"/"figuras_articulo"; OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "axes.titlesize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 13, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 100})
C = dict(real="#222222", et="#0072B2", rf="#009E73", gb="#E69F00", lgb="#CC79A7", ridge="#D55E00", dummy="#999999")
R = 100.0
FE = ['xang','velU','velV','air_dens','HR','hour','xmag_disc','TEMP_norm']



ALT = {"cajica": 2558.0, "jpa": 40.0}
def air_density(t, h): return 1.225 * (288.0 / (t + 273.0)) * np.exp(-h / 8435.0)
def site_features(df, site):
    th = np.deg2rad(df["xang"]); o = pd.DataFrame(index=df.index)
    o["xang"] = df["xang"]; o["velU"] = df["xmag"]*np.cos(th); o["velV"] = df["xmag"]*np.sin(th)
    o["air_dens"] = air_density(df["T"], ALT[site]); o["HR"] = df["HR"]; o["hour"] = df.index.hour; o["xmag"] = df["xmag"]
    o["xmag_disc"] = df["xmag"].where(df["xmag"] >= 3.5, 0.0); o["TEMP_norm"] = df["T"].where(df["T"] >= 23, 0.0); return o
COLS = ['ID','velU','velV','velW','xang','xejz','xmag','V_G1','FR_G1','P_AT','EG1','V_G2','FR_G2','P_AT2','EG2','HUMEDAD_SUELO','HUMEDAD_RELATIVA','TEMPERATURA','fecha']
df = pd.read_csv(P/"data/data_cajica_2022_crudo.csv", names=COLS, header=None, parse_dates=['fecha'])
df.loc[df.xmag > 65, 'xmag'] = 0; df.loc[df.V_G1 > 25, 'V_G1'] = 0
df.loc[df.HUMEDAD_RELATIVA > 70, 'HUMEDAD_RELATIVA'] = df.HUMEDAD_RELATIVA.mean(); df.loc[df.HUMEDAD_RELATIVA < 30, 'HUMEDAD_RELATIVA'] = df.HUMEDAD_RELATIVA.mean()
real = df.drop_duplicates('fecha', keep='last').sort_values('fecha').set_index('fecha')
grid = real.asfreq('60s', method='bfill'); grid['is_real'] = grid.index.isin(real.index)
tr = grid.rename(columns={'HUMEDAD_RELATIVA':'HR','TEMPERATURA':'T'})
M = pd.concat([site_features(tr,'cajica'), tr[['V_G1','is_real']]], axis=1).dropna()
X, y, isr = M[FE], M.V_G1, M.is_real.values
folds = list(TimeSeriesSplit(n_splits=10, gap=1440).split(X))
print(len(M), isr.sum(), "R2 esperado 0.4382")

MODELS = {"Extra Trees": (lambda: ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=123), C["et"]),
          "Random Forest": (lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=123), C["rf"]),
          "Gradient Boosting": (lambda: GradientBoostingRegressor(random_state=123), C["gb"]),
          "LightGBM": (lambda: LGBMRegressor(random_state=123, verbose=-1), C["lgb"]),
          "Ridge": (lambda: Ridge(), C["ridge"]), "Dummy (media)": (lambda: DummyRegressor(), C["dummy"])}
oof = {k: pd.Series(np.nan, index=M.index) for k in MODELS}; fold_r2 = []; imp_rows = []
rng = np.random.default_rng(0)
for fi, (a, b) in enumerate(folds):
    bb = b[isr[b]]; Xt, yt = X.iloc[bb], y.iloc[bb]
    for k, (mk, _) in MODELS.items():
        m = mk().fit(X.iloc[a], y.iloc[a]); p = m.predict(Xt); oof[k].iloc[bb] = p
        if k == "Extra Trees":
            fold_r2.append(r2_score(yt, p)); base = mean_absolute_error(yt, p); row = {}
            for f in FE:
                dm = []
                for _ in range(5):
                    Xp = Xt.copy(); Xp[f] = rng.permutation(Xp[f].values); dm.append(mean_absolute_error(yt, m.predict(Xp)) - base)
                row[f] = np.mean(dm)
            imp_rows.append(row)
print("R2 por fold ET", np.round(fold_r2,3), "media", np.mean(fold_r2))
imp = pd.DataFrame(imp_rows); imp.to_csv(OUT/"importancia_permutacion_dMAE.csv", index=False)
ev = ~oof["Extra Trees"].isna(); oofET = oof["Extra Trees"][ev]; yy = y[ev]
print("pooled R2", r2_score(yy, oofET), "MAE", mean_absolute_error(yy, oofET), "periodo", yy.index.min(), yy.index.max())
pd.DataFrame(oof).assign(real=y).dropna(subset=["Extra Trees"]).to_csv(OUT/"predicciones_cronologicas_cajica.csv")

NAMES = {"xang":"Wind angle (°)","velU":"Wind $U$ component (m/s)","velV":"Wind $V$ component (m/s)","air_dens":"Air density (kg/m³)",
         "HR":"Relative humidity (%)","hour":"Hour of day (h)","xmag_disc":"Wind speed ≥ 3.5 m/s (m/s)","TEMP_norm":"Temperature ≥ 23 °C (°C)"}

# ---- Fig 1a: real vs predicho + residuos
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
hb = ax[0].hexbin(yy, oofET, gridsize=45, bins="log", cmap="viridis", mincnt=1, extent=(0, 20, 0, 20))
lim = max(yy.max(), oofET.max()); ax[0].plot([0, lim], [0, lim], "r--", lw=2, label="Ideal (y = x)")
ax[0].set_xlabel("Measured $V_{RMS}$ (V)"); ax[0].set_ylabel("Predicted $V_{RMS}$ (V)"); ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
ax[0].legend(loc="upper left", title=f"Pooled test folds\n$R^2$ = {r2_score(yy,oofET):.2f}, MAE = {mean_absolute_error(yy,oofET):.2f} V")
cb = fig.colorbar(hb, ax=ax[0]); cb.set_label("Records (log scale)")
res = yy - oofET
ax[1].hexbin(oofET, res, gridsize=45, bins="log", cmap="viridis", mincnt=1); ax[1].axhline(0, color="r", ls="--", lw=2)
ax[1].set_xlabel("Predicted $V_{RMS}$ (V)"); ax[1].set_ylabel("Residual (measured − predicted) (V)")
plt.tight_layout(); fig.savefig(OUT/"fig_prediction_error.png"); plt.close()

# ---- Fig 1b: importancia por permutacion
m_ = imp.mean().sort_values(); s_ = imp.std().reindex(m_.index)
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.barh([NAMES[i] for i in m_.index], m_.values, xerr=s_.values, color=C["et"], ecolor="#222", capsize=4)
ax.set_xlabel("Increase in MAE after permuting the feature (V)\nmean ± s.d. over 10 chronological folds"); ax.grid(axis="y", alpha=0)
plt.tight_layout(); fig.savefig(OUT/"fig_feature_importance.png"); plt.close()
print("importancia dMAE:\n", m_.round(4).to_string())

