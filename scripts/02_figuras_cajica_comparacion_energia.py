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




d = pd.read_csv(OUT/"predicciones_cronologicas_cajica.csv", parse_dates=[0], index_col=0)
def multi(t0, t1, name):
    s = d[t0:t1]; fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(s.index, s["Dummy (media)"], color=C["dummy"], lw=1.5, ls="--", label="Mean predictor")
    ax.plot(s.index, s["Gradient Boosting"], color=C["gb"], lw=1.6, label="Gradient Boosting")
    ax.plot(s.index, s["Extra Trees"], color=C["et"], lw=1.6, label="Extra Trees")
    ax.plot(s.index, s["real"], color=C["real"], lw=1.2, alpha=.85, label="Measured", zorder=1)
    ax.set_ylabel("$V_{RMS}$ (V)"); ax.set_xlabel("Date and time (2022)"); ax.set_ylim(-0.5, None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M")); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.15), frameon=False)
    plt.tight_layout(); fig.savefig(OUT/name); plt.close()
multi("2022-08-14", "2022-08-20", "fig_cajica_multi.png"); multi("2022-08-15 12:00", "2022-08-16 06:00", "fig_cajica_detail.png")

a = d["2022-08-01":"2022-08-20"]; a_y, a_p = a["real"], a["Extra Trees"]
fig, ax = plt.subplots(2, 1, figsize=(13, 7.8), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
ax[0].plot(a_y.index, a_y.values, color=C["real"], lw=1.2, label="Measured"); ax[0].plot(a_p.index, a_p.values, color=C["et"], lw=1.2, label="Extra Trees (chronological test)")
ax[0].set_ylabel("$V_{RMS}$ (V)"); ax[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False)
Ep = (a_p**2/R/60).resample("10min").sum(); Ey = (a_y**2/R/60).resample("10min").sum()
ax[1].plot(Ey.index, Ey.values, color=C["real"], lw=1.2); ax[1].plot(Ep.index, Ep.values, color=C["et"], lw=1.2)
ax[1].set_ylabel("Energy per 10 min (Wh)"); ax[1].set_xlabel("Date (2022)"); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.tight_layout(); fig.savefig(OUT/"fig_cajica_vrms.png"); plt.close()
print(Ey.sum(), Ep.sum())
