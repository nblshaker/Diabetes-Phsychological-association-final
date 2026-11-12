"""Extra analyses requested in supervisor review (non-private anchor, AP, ablation, DP re-run at one lr, sign change, probabilities)."""
import json, os, importlib.util, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr, rankdata
import statsmodels.api as sm
BASE=os.path.dirname(os.path.abspath(__file__))
import dp_accountant as acct
spec=importlib.util.spec_from_file_location("dp","02b_dp_federated_learning.py"); dp=importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)
fl=dp.fl02; CL=dp.CLIENTS; sig=dp.sigmoid
LR,C,B=0.15,0.5,1024
OUT={}
def load(feats,label):
    dfs=[fl.load_client(c) for c in CL]
    mean,std=fl.federated_feature_stats(dfs,feats)
    rng=np.random.RandomState(7);out=[]
    for d in dfs:
        idx=np.arange(len(d));rng.shuffle(idx);nt=int(len(d)*0.2)
        te,tr=d.iloc[idx[:nt]],d.iloc[idx[nt:]]
        f=lambda s:(((s[feats].values-mean)/std),s[label].values.astype(float))
        out.append({"train":f(tr),"test":f(te)})
    return out
def ev(cl,w,b):
    per={};P=[];Y=[]
    for n,c in zip(CL,cl):
        X,y=c["test"];p=sig(X@w+b)
        per[n]={"auc":dp.auc_rank(p,y),"ap":float(average_precision_score(y,p))};P.append(p);Y.append(y)
    P=np.concatenate(P);Y=np.concatenate(Y)
    return {"per":per,"auc":dp.auc_rank(P,Y),"ap":float(average_precision_score(Y,P)),"prev":float(Y.mean())}
def local_only(cl):
    res=[]
    P=[];Y=[];per={}
    for n,c in zip(CL,cl):
        Xtr,ytr=c["train"];w,b=dp.local_train(Xtr,ytr,np.zeros(Xtr.shape[1]),0.0,375,LR,1.0,0.0,np.inf,None)
        X,y=c["test"];p=sig(X@w+b);per[n]={"auc":dp.auc_rank(p,y),"ap":float(average_precision_score(y,p))};P.append(p);Y.append(y)
    P=np.concatenate(P);Y=np.concatenate(Y)
    return {"per":per,"auc":dp.auc_rank(P,Y),"ap":float(average_precision_score(Y,P))}
F_DEP=fl.FEATS_DEPRESSION;F_DIA=fl.FEATS_DIABETES
specs={"depression":(F_DEP,"depression_binary",[f for f in F_DEP if f!="LBXGH"]),
       "diabetes":(F_DIA,"diabetes_binary",[f for f in F_DIA if f!="PHQ9_Total"])}
for t,(feats,lab,abl) in specs.items():
    cl=load(feats,lab);R={}
    w,b,_=dp.fedavg(cl,LR,None,np.inf,None);R["fed"]=ev(cl,w,b);R["local"]=local_only(cl)
    # operating point from TRAINING predictions only (Youden J on pooled training predictions)
    Ptr=np.concatenate([sig(c["train"][0]@w+b) for c in cl]);Ytr=np.concatenate([c["train"][1] for c in cl])
    best=None
    for th in np.quantile(Ptr,np.linspace(0.05,0.99,200)):
        pr=Ptr>=th;se=(pr&(Ytr==1)).sum()/(Ytr==1).sum();sp=((~pr)&(Ytr==0)).sum()/(Ytr==0).sum()
        if best is None or se+sp>best[0]:best=(se+sp,th,se,sp)
    th=best[1];Pte=np.concatenate([sig(c["test"][0]@w+b) for c in cl]);Yte=np.concatenate([c["test"][1] for c in cl])
    pr=Pte>=th
    R["operating_point"]={"threshold":float(th),"train_sens":float(best[2]),"train_spec":float(best[3]),
        "test_sens":float((pr&(Yte==1)).sum()/(Yte==1).sum()),"test_spec":float(((~pr)&(Yte==0)).sum()/(Yte==0).sum()),
        "test_acc":float((pr==Yte).mean()),"test_acc_at_0.5":float(((Pte>=0.5)==Yte).mean()),"pos_rate_at_0.5":float((Pte>=0.5).mean())}
    # ablation: cross-domain score removed
    cla=load(abl,lab);w2,b2,_=dp.fedavg(cla,LR,None,np.inf,None);R["ablation_fed"]=ev(cla,w2,b2);R["ablation_local"]=local_only(cla)
    R["n_train"]=[len(c["train"][1]) for c in cl];R["n_test"]=[len(c["test"][1]) for c in cl]
    R["steps_per_pass"]=[len(c["train"][1])/B for c in cl];R["q"]=[B/len(c["train"][1]) for c in cl]
    # DP at lr=0.15, C, B
    R["mb_noclip"]=ev(cl,*dp.fedavg(cl,LR,B,np.inf,None,seed=0)[:2])
    runs=[ev(cl,*dp.fedavg(cl,LR,B,C,None,seed=s)[:2]) for s in range(20)];R["mb_clip"]=runs
    for eps in [0.5,1,2,4,8,16]:
        runs=[];
        for s in range(20):
            w3,b3,info=dp.fedavg(cl,LR,B,C,float(eps),seed=s);runs.append(ev(cl,w3,b3))
        R[f"dp{eps}"]={"runs":runs,"sigma":info["sigma"],"q":info["q"],"eps_total":info["eps"],
            "eps_single":[acct.epsilon(q,sg,375,1e-5) for q,sg in zip(info["q"],info["sigma"])]}
    OUT[t]=R;print(t,"done",flush=True)
json.dump(OUT,open("results/review_models.json","w"),indent=1)
# ---- sign change
def pz(rs,ns):
    z=np.arctanh(rs);v=1/(np.array(ns)-3);w=1/v;zb=(w*z).sum()/w.sum();se=(1/w.sum())**.5
    from scipy.stats import norm
    return float(np.tanh(zb)),float(np.tanh(zb-1.96*se)),float(np.tanh(zb+1.96*se)),float(2*norm.sf(abs(zb/se)))
def partial_sp(x,y,Z):
    rx,ry=rankdata(x),rankdata(y);Zr=np.column_stack([np.ones(len(x))]+[rankdata(z) for z in Z.T])
    ex=rx-Zr@np.linalg.lstsq(Zr,rx,rcond=None)[0];ey=ry-Zr@np.linalg.lstsq(Zr,ry,rcond=None)[0]
    return float(np.corrcoef(ex,ey)[0,1])
S={"unadj":[],"adj_age_sex":[],"adj_all5":[],"harm_unadj":[],"harm_adj_age_sex":[],"n":[]}
items=["DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090"]
chk=[]
for c in CL:
    d=fl.load_client(c);n=len(d);S["n"].append(n)
    S["unadj"].append(float(spearmanr(d.LBXGH,d.PHQ9_Total)[0]))
    S["adj_age_sex"].append(partial_sp(d.LBXGH.values,d.PHQ9_Total.values,d[["RIDAGEYR","RIAGENDR"]].values))
    Z5=d[["RIDAGEYR","RIAGENDR","DMDEDUC2","INDFMPIR","BMXBMI"]]
    S["adj_all5"].append(partial_sp(d.LBXGH.values,d.PHQ9_Total.values,Z5.values))
    it=d[items];valid=it.isin([0,1,2,3]).all(axis=1);h=it.sum(axis=1).where(valid)
    chk.append({"client":c,"items_valid_frac":float(valid.mean()),"total_eq_item_sum_frac":float((d.PHQ9_Total==it.sum(axis=1)).mean()),"mean_total":float(d.PHQ9_Total.mean()),"mean_item_sum":float(it.sum(axis=1).mean()),"max_total":float(d.PHQ9_Total.max()),"n_valid":int(valid.sum())})
    m=valid.values
    S["harm_unadj"].append(float(spearmanr(d.LBXGH[m],h[m])[0]))
    S["harm_adj_age_sex"].append(partial_sp(d.LBXGH.values[m],h.values[m],d[["RIDAGEYR","RIAGENDR"]].values[m]))
    S.setdefault("harm_n",[]).append(int(m.sum()))
S["pooled_unadj"]=pz(S["unadj"],S["n"]);S["pooled_adj_age_sex"]=pz(S["adj_age_sex"],S["n"]);S["pooled_adj_all5"]=pz(S["adj_all5"],S["n"])
S["pooled_harm_unadj"]=pz(S["harm_unadj"],S["harm_n"]);S["pooled_harm_adj_age_sex"]=pz(S["harm_adj_age_sex"],S["harm_n"])
S["phq_check"]=chk
# ---- predicted probabilities from per-site adjusted models, pooled by inverse variance
dfs=[fl.load_client(c) for c in CL];N=sum(len(d) for d in dfs)
def fit_pool(xcols,lab):
    bs=[];vs=[]
    for d in dfs:
        X=sm.add_constant(d[xcols]);r=sm.Logit(d[lab],X).fit(disp=0);bs.append(r.params.values);vs.append(r.bse.values**2)
    bs=np.array(bs);vs=np.array(vs);w=1/vs;return (w*bs).sum(0)/w.sum(0),xcols
P={}
allx=pd.concat(dfs)
for lab,key,cols,lo,step in [("diabetes_binary","PHQ9_Total",["PHQ9_Total","RIAGENDR","RIDAGEYR","DMDEDUC2","INDFMPIR","BMXBMI"],None,10),
                              ("depression_binary","LBXGH",["LBXGH","RIAGENDR","RIDAGEYR","DMDEDUC2","INDFMPIR","BMXBMI"],None,1)]:
    beta,_=fit_pool(cols,lab)
    base={c:float(allx[c].median()) for c in cols};base["RIAGENDR"]=2.0
    def prob(v):
        x=dict(base);x[key]=v;z=beta[0]+sum(beta[i+1]*x[c] for i,c in enumerate(cols));return float(1/(1+np.exp(-z)))
    b0=base[key];P[lab]={"base":base,"beta_key":float(beta[1]),"p_base":prob(b0),"p_plus":prob(b0+step),"step":step,
        "OR_step":float(np.exp(beta[1]*step)),"p_base_male":None}
    base2=dict(base);base2["RIAGENDR"]=1.0
    x=dict(base2);z0=beta[0]+sum(beta[i+1]*x[c] for i,c in enumerate(cols));x[key]=b0+step;z1=beta[0]+sum(beta[i+1]*x[c] for i,c in enumerate(cols))
    P[lab]["p_base_male"]=float(1/(1+np.exp(-z0)));P[lab]["p_plus_male"]=float(1/(1+np.exp(-z1)))
    # unadjusted OR per step too
S["probabilities"]=P
json.dump(S,open("results/review_signchange.json","w"),indent=1);print(json.dumps(S,indent=1))
