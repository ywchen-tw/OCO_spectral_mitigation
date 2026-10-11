"""Sample the HYSPLIT column (0-5000 m) concentration grid at OCO-2 footprints and compare with the observed enhancement.
Run after run.sh:  conda run -n ml310 python postprocess.py
"""
import glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LAT0,LON0=38.2792,-89.6669; CONV=0.0154   # kg CO2 m-2 per ppm XCO2
R="/Users/yuch8913/programming/oco_fp_analysis/results/model_comparison/deep_ensemble/de_beta_nll_prof_reg_foldpca_o05l15_m5/us_coal_plumes/combined_2019-09-14/plot_data.parquet"
# --- read con2asc output (one file per sampling period): columns DAY HR LAT LON CO2_conc (mass/m3 averaged over the 0-5000 m layer)
files=sorted(glob.glob("cdump_*"))
files=[f for f in files if not f.endswith(".ps")]
frames=[]
for f in files:
    try: d=pd.read_csv(f,sep=r"\s+",engine="python"); frames.append(d)
    except Exception as e: print("skip",f,e)
g=pd.concat(frames); g.columns=[c.strip().upper() for c in g.columns]
cc=[c for c in g.columns if "CO2" in c][0]
g["col_kgm2"]=g[cc]*5000.0; g["dxco2_ppm"]=g.col_kgm2/CONV
print("grid cells",len(g),"max ΔXCO2 %.2f ppm"%g.dxco2_ppm.max())
# --- OCO-2 footprints near the plant
t=pd.read_parquet(R,columns=["lat","lon","xco2_bc","deep_ensemble_corrected_xco2","cld_dist_km"])
t=t[(np.abs(t.lat-LAT0)<0.5)&(np.abs(t.lon-LON0)<0.5)].copy(); t["s"]=(t.lat-LAT0)*111.0
bg=t[np.abs(t.s)>20]; t["e_bc"]=t.xco2_bc-bg.xco2_bc.median(); t["e_corr"]=t.deep_ensemble_corrected_xco2-bg.deep_ensemble_corrected_xco2.median()
# nearest grid cell (0.01 deg)
from scipy.spatial import cKDTree
tree=cKDTree(np.c_[g.LAT,g.LON]); d,i=tree.query(np.c_[t.lat,t.lon]); t["hysplit_ppm"]=np.where(d<0.015,g.dxco2_ppm.to_numpy()[i],0.0)
t=t.sort_values("s"); t.to_csv("footprints_vs_hysplit.csv",index=False)
b=t.groupby(np.round(t.s/2)*2)[["e_bc","e_corr","hysplit_ppm"]].median()
print(b.loc[-12:12].round(2))
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].scatter(t.s,t.e_bc,s=8,color="#2a78d6",alpha=.4); ax[0].plot(b.index,b.e_bc,color="#2a78d6",lw=2,label="observed, bias-corrected")
ax[0].scatter(t.s,t.e_corr,s=8,color="#eb6834",alpha=.4); ax[0].plot(b.index,b.e_corr,color="#eb6834",lw=2,label="observed, DE-corrected")
ax[0].plot(b.index,b.hysplit_ppm,color="#1baf7a",lw=2,label="HYSPLIT column (CAMPD emission)"); ax[0].axvline(0,color=".5",ls="--"); ax[0].set_xlim(-30,30); ax[0].set_xlabel("along-track km (+north)"); ax[0].set_ylabel("ΔXCO2 (ppm)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.2)
sc=ax[1].scatter(g.LON,g.LAT,c=g.dxco2_ppm,s=4,cmap="Reds",vmin=0,vmax=max(1,g.dxco2_ppm.quantile(.99))); ax[1].scatter(t.lon,t.lat,c=t.e_bc,s=10,cmap="RdBu_r",vmin=-3,vmax=3,edgecolor="k",lw=.2)
ax[1].plot(LON0,LAT0,"*",color="#ffd400",ms=12,mec="k"); ax[1].set_xlim(LON0-.5,LON0+.5); ax[1].set_ylim(LAT0-.5,LAT0+.5); ax[1].set_title("HYSPLIT column ΔXCO2 (red) + OCO-2 footprints"); plt.colorbar(sc,ax=ax[1],label="ppm")
fig.tight_layout(); fig.savefig("hysplit_vs_oco2.png",dpi=120); print("saved hysplit_vs_oco2.png")
