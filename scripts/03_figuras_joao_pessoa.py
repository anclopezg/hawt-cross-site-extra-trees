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



# ---- JPA
jp = pd.read_csv(P/"outputs"/"predicciones_jpa_tablaIV.csv", parse_dates=["fecha"], index_col="fecha")
dt_h = jp.index.to_series().diff().median().total_seconds()/3600; print("JPA", len(jp), jp.index.min(), jp.index.max(), dt_h*60, "min")
def jfig(s, name, fmt, xl, w=13, extra=None):
    fig, ax = plt.subplots(figsize=(w, 5.2)); ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax.plot(s.index, s.xmag, color="#4a6fa5", lw=1.2, label="Wind speed (m/s)"); ax.axhline(2.5, color="#4a6fa5", ls=":", lw=1.5, label="Cut-in 2.5 m/s")
    ax2.fill_between(s.index, s.V_p10, s.V_p90, color="#009E73", alpha=.25, label="Tree spread (p10–p90)")
    ax2.plot(s.index, s.V_pred, color="#009E73", lw=1.4, label="Predicted $V_{RMS}$ (V)")
    ax.set_ylabel("Wind speed (m/s)", color="#4a6fa5"); ax2.set_ylabel("Predicted $V_{RMS}$ (V)", color="#009E73"); ax2.grid(False); ax.set_xlabel(xl)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1+h2, l1+l2, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.22), frameon=False)
    plt.tight_layout(); fig.savefig(OUT/name); plt.close()
jpAM = jp["2025-04-01":"2025-05-31 23:59"]; print("AprMay n", len(jpAM), jpAM.V_pred.max())
jfig(jpAM, "fig_jp_full.png", "%b %d", "Date (2025)")
jfig(jp["2025-05-12":"2025-05-16 23:59"], "fig_jp_zoom.png", "%b %d\n%H:%M", "Date and time (2025)")
z = jp["2025-05-12":"2025-05-15 23:59"]; print("zoom n", len(z), "Vmax", z.V_pred.max())
fig, ax = plt.subplots(figsize=(13, 5.2)); ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax.fill_between(z.index, z.V_p10**2/R, z.V_p90**2/R, color="#E69F00", alpha=.3, label="Tree spread (p10–p90)"); ax.plot(z.index, z.P_W, color="#E69F00", lw=1.5, label="Power (W)")
ax2.plot(z.index, z.E_Wh.cumsum(), color="#7a3b8f", lw=2, label="Cumulative energy (Wh)"); ax2.grid(False)
ax.set_ylabel("Power $V_{RMS}^2/R$ (W)"); ax2.set_ylabel("Cumulative energy (Wh)", color="#7a3b8f"); ax.set_xlabel("Date and time (2025)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1+h2, l1+l2, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.18), frameon=False)
plt.tight_layout(); fig.savefig(OUT/"fig_jp_energy.png"); plt.close()
print("energia acumulada 12-15 mayo (Wh):", z.E_Wh.sum(), "P max", z.P_W.max())
