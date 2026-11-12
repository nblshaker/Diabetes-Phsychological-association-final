"""Repeat the federated-vs-local comparison over many random 80/20 splits (same settings as Section 4.4.5)."""
import json, os, importlib.util, numpy as np
from sklearn.metrics import average_precision_score
BASE=os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
spec=importlib.util.spec_from_file_location("dp","02b_dp_federated_learning.py"); dp=importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)
fl=dp.fl02; CL=dp.CLIENTS; sig=dp.sigmoid; LR=0.15; NS=200
dfs=[fl.load_client(c) for c in CL]
def load(feats,label,seed,mean,std):
    rng=np.random.RandomState(seed); out=[]
    for d in dfs:
        idx=np.arange(len(d)); rng.shuffle(idx); nt=int(len(d)*0.2)
        f=lambda s:(((s[feats].values-mean)/std),s[label].values.astype(float))
        out.append({"train":f(d.iloc[idx[nt:]]),"test":f(d.iloc[idx[:nt]])})
    return out
def ev(cl,pred):
    P=[];Y=[];per=[]
    for c,(w,b) in zip(cl,pred):
        X,y=c["test"];p=sig(X@w+b);P.append(p);Y.append(y);per.append(dp.auc_rank(p,y))
    P=np.concatenate(P);Y=np.concatenate(Y)
    return dp.auc_rank(P,Y),float(average_precision_score(Y,P)),per
R={}
for task,feats,lab in (("depression",fl.FEATS_DEPRESSION,"depression_binary"),("diabetes",fl.FEATS_DIABETES,"diabetes_binary")):
    mean,std=fl.federated_feature_stats(dfs,feats); rows=[]
    for s in range(NS):
        cl=load(feats,lab,1000+s,mean,std)
        w,b,_=dp.fedavg(cl,LR,None,np.inf,None)
        fa,fap,fper=ev(cl,[(w,b)]*3)
        loc=[]
        for c in cl:
            X,y=c["train"];lw,lb=dp.local_train(X,y,np.zeros(X.shape[1]),0.0,375,LR,1.0,0.0,np.inf,None);loc.append((lw,lb))
        la,lap,lper=ev(cl,loc)
        rows.append({"fed_auc":fa,"loc_auc":la,"fed_ap":fap,"loc_ap":lap,"fed_per":fper,"loc_per":lper})
    R[task]=rows; print(task,"done",flush=True)
def ci(x): x=np.asarray(x); return [float(np.mean(x)),float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]
S={}
for t,rows in R.items():
    gap=[r["loc_auc"]-r["fed_auc"] for r in rows]; gapap=[r["loc_ap"]-r["fed_ap"] for r in rows]
    S[t]={"n_splits":NS,"fed_auc":ci([r["fed_auc"] for r in rows]),"loc_auc":ci([r["loc_auc"] for r in rows]),"gap_auc":ci(gap),
          "fed_ap":ci([r["fed_ap"] for r in rows]),"loc_ap":ci([r["loc_ap"] for r in rows]),"gap_ap":ci(gapap),
          "frac_local_higher_auc":float(np.mean(np.array(gap)>0)),"frac_local_higher_ap":float(np.mean(np.array(gapap)>0)),
          "per_hospital_gap_auc":[ci([r["loc_per"][k]-r["fed_per"][k] for r in rows]) for k in range(3)],
          "per_hospital_frac_local_higher":[float(np.mean([r["loc_per"][k]>r["fed_per"][k] for r in rows])) for k in range(3)]}
json.dump(S,open("results/repeated_splits.json","w"),indent=1); print(json.dumps(S,indent=1))
