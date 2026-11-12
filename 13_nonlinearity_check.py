"""Non-linearity check for the PHQ-9 / HbA1c relationship (supervisor request).
Per site: likelihood-ratio test of adding squared + cubic terms (2 df) to the linear term, for diabetes ~ PHQ-9 and
depression ~ HbA1c (logistic), and an F test for log(HbA1c) ~ PHQ-9 (OLS). Unadjusted and adjusted for age + sex.
Predictors are standardised within site for numerical stability. Only test statistics leave a site; the coordinator
sums the per-site chi-square values (df adds). Plots: binned means with 95% CI and a LOWESS curve."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=["Client_A_2015-2016","Client_B_2017-2020","Client_C_2021-2023"]
D={s:pd.read_csv(f"data/{s}.csv") for s in S}
def design(d,x,poly,adj):
    z=(d[x]-d[x].mean())/d[x].std(); cols={"z":z}
    if poly: cols["z2"]=z**2; cols["z3"]=z**3
    if adj: cols["age"]=(d.RIDAGEYR-d.RIDAGEYR.mean())/d.RIDAGEYR.std(); cols["female"]=(d.RIAGENDR==2).astype(float)
    return sm.add_constant(pd.DataFrame(cols))
def lr(d,y,x,adj):
    m0=sm.Logit(d[y],design(d,x,False,adj)).fit(disp=0); m1=sm.Logit(d[y],design(d,x,True,adj)).fit(disp=0)
    return 2*(m1.llf-m0.llf),2
out={}
for name,(y,x) in {"diabetes ~ PHQ-9":("diabetes_binary","PHQ9_Total"),"depression ~ HbA1c":("depression_binary","LBXGH")}.items():
    for adj in (False,True):
        chis=[];per=[]
        for s in S:
            c,df=lr(D[s],y,x,adj); chis.append(c); per.append((round(c,2),df,round(float(stats.chi2.sf(c,df)),4)))
        out[f"{name} | {'adjusted' if adj else 'unadjusted'}"]={"per_site(chi2,df,p)":per,"pooled_chi2":sum(chis),"pooled_df":2*len(S),"pooled_p":float(stats.chi2.sf(sum(chis),2*len(S)))}
for adj in (False,True):
    r=[]
    for s in S:
        d=D[s]; yv=np.log(d.LBXGH)
        m0=sm.OLS(yv,design(d,"PHQ9_Total",False,adj)).fit(); m1=sm.OLS(yv,design(d,"PHQ9_Total",True,adj)).fit()
        f,p,df=m1.compare_f_test(m0); r.append((round(float(f),2),round(float(p),4),int(df)))
    out[f"log(HbA1c) ~ PHQ-9 (OLS) | {'adjusted' if adj else 'unadjusted'}"]={"per_site(F,p,df)":r}
json.dump(out,open("results/nonlinearity.json","w"),indent=1)
for k,v in out.items(): print(k,json.dumps(v))
cols=['#2B5C8A','#4F81AA','#7FAF9B']; labs=["Hospital A (2015-2016)","Hospital B (2017-2020)","Hospital C (2021-2023)"]
fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True); bins=[-1,0,1,2,3,4,6,9,14,30]
for ax,s,c,lab in zip(axs,S,cols,labs):
    d=D[s].copy(); d["b"]=pd.cut(d.PHQ9_Total,bins); g=d.groupby("b",observed=True)
    ax.errorbar(g.PHQ9_Total.mean(),g.LBXGH.mean(),yerr=1.96*g.LBXGH.sem(),fmt='o',color=c,capsize=2,label='Mean HbA1c in PHQ-9 band (95% CI)')
    lo=lowess(d.LBXGH,d.PHQ9_Total,frac=.5,it=0); ax.plot(lo[:,0],lo[:,1],color='black',lw=1.4,label='LOWESS curve')
    ax.set_title(lab,fontsize=10); ax.set_xlabel('PHQ-9 total'); ax.grid(alpha=.3); ax.set_xlim(-0.5,22); ax.set_ylim(5.3,6.4)
axs[0].set_ylabel('HbA1c (%)'); axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("charts/10_nonlinearity_hba1c_by_phq9.png",dpi=170)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for s,c,lab in zip(S,cols,["Hospital A","Hospital B","Hospital C"]):
    d=D[s].copy(); d["b"]=pd.cut(d.PHQ9_Total,[-1,0,2,4,9,14,30]); g=d.groupby("b",observed=True)
    axs[0].errorbar(g.PHQ9_Total.mean(),g.diabetes_binary.mean()*100,yerr=1.96*g.diabetes_binary.sem()*100,fmt='-o',color=c,capsize=2,label=lab)
    d["h"]=pd.qcut(d.LBXGH,8,duplicates='drop'); g=d.groupby("h",observed=True)
    axs[1].errorbar(g.LBXGH.mean(),g.depression_binary.mean()*100,yerr=1.96*g.depression_binary.sem()*100,fmt='-o',color=c,capsize=2,label=lab)
axs[0].set_xlabel('PHQ-9 total (band mean)'); axs[0].set_ylabel('Diabetes prevalence, HbA1c ≥ 6.5% (%)'); axs[1].set_xlabel('HbA1c % (octile mean)'); axs[1].set_ylabel('Depression prevalence, PHQ-9 ≥ 10 (%)')
for a in axs: a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("charts/11_nonlinearity_prevalence_bands.png",dpi=170)
