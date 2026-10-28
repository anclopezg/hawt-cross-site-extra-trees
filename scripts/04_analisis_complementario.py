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


from sklearn.metrics import precision_score, recall_score, f1_score
rows=[]; cover=[]; preds=[]
for a,b in folds:
    bb=b[isr[b]]; m=ExtraTreesRegressor(n_estimators=100,n_jobs=-1,random_state=123).fit(X.iloc[a],y.iloc[a])
    per=np.stack([t.predict(X.iloc[bb].values) for t in m.estimators_]); p=per.mean(0); lo,hi=np.percentile(per,[10,90],axis=0)
    yt=y.iloc[bb].values
    preds.append(pd.DataFrame({"y":yt,"p":p,"lo":lo,"hi":hi},index=X.index[bb]))
D=pd.concat(preds); D.to_csv(OUT/"predicciones_cronologicas_ET_bandas.csv")
inside=(D.y>=D.lo)&(D.y<=D.hi); print("cobertura p10-p90 total %.3f | y>=1V %.3f | y<1V %.3f | ancho medio %.3f"%(inside.mean(), inside[D.y>=1].mean(), inside[D.y<1].mean(), (D.hi-D.lo).mean()))
for thr in (0.5,1.0,2.0):
    ev=D.y>=thr; pe=D.p>=thr
    print("evento V>=%.1f: prevalencia %.3f precision %.3f recall %.3f F1 %.3f"%(thr, ev.mean(), precision_score(ev,pe), recall_score(ev,pe), f1_score(ev,pe)))
E=pd.DataFrame({"Em":D.y**2/R/60,"Ep":D.p**2/R/60,"Vm":D.y,"Vp":D.p})
for fr,name in (("10min","10 min"),("1h","hora"),("1D","dia")):
    g=E.resample(fr).agg({"Em":"sum","Ep":"sum","Vm":"mean","Vp":"mean"}); g=g[(E.Em.resample(fr).count()>=0.5*({"10min":10,"1h":60,"1D":1440}[fr]))]
    print("%-7s n=%4d  R2 energia %.3f  R2 Vmedio %.3f  sesgo energia %.2f"%(name,len(g),r2_score(g.Em,g.Ep),r2_score(g.Vm,g.Vp),g.Ep.sum()/g.Em.sum()))
# --- contraste fisico en JPA
def Pphys(v,rho,A=0.636,Cp=0.35,ci=2.5,co=15.0):
    p=0.5*rho*A*Cp*v**3; return np.where((v<ci)|(v>co),0.0,p)
Mr=M[isr]; Pc=Pphys(Mr.xmag.values, Mr.air_dens.values); eta=(Mr.V_G1**2).sum()/(Pc*R).sum(); print("eta (energia) %.4f"%eta)
jp=pd.read_csv(P/"outputs"/"predicciones_jpa_tablaIV.csv",parse_dates=["fecha"],index_col="fecha")["2025-04-01":"2025-05-31 23:59"]
rho=air_density(jp["T"],40.0); Pj=Pphys(jp.xmag.values,rho); Vb=np.sqrt(eta*Pj*R)
Eet=(jp.V_pred**2/R/6).sum(); Eb=(Vb**2/R/6).sum()
print("JPA abr-may: ET %.2f Wh | fisico %.2f Wh | V max ET %.2f fisico %.2f | frac registros con viento>=2.5: %.3f"%(Eet,Eb,jp.V_pred.max(),Vb.max(),(jp.xmag>=2.5).mean()))
# Cajica misma comparacion de energia en test
Pt=Pphys(X.loc[D.index,'velU'].values*0+np.hypot(X.loc[D.index,'velU'],X.loc[D.index,'velV']).values, X.loc[D.index,'air_dens'].values)
print("Cajica test: energia medida %.2f | ET %.2f | fisico %.2f Wh"%((D.y**2/R/60).sum(),(D.p**2/R/60).sum(),(eta*Pt/60).sum()))
