"""Step 11: (1) pooled (centralized) baseline vs federated vs local-only, 200 random 80/20 splits;
          (2) small-site experiment: shrink one hospital's training data and compare federated vs local-only on that hospital's test set.
Same model, features, learning rate (0.15), 15 rounds x 25 steps = 375 gradient steps, federated standardisation as in Sections 4.4.5 / 10_repeated_splits.py."""
import json, os, importlib.util, numpy as np
from sklearn.metrics import average_precision_score
BASE=os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
spec=importlib.util.spec_from_file_location("dp","02b_dp_federated_learning.py"); dp=importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)
fl=dp.fl02; CL=dp.CLIENTS; sig=dp.sigmoid; LR=0.15; STEPS=375; NS=200; NR=100
dfs=[fl.load_client(c) for c in CL]; NAMES=["Site 1","Site 2","Site 3"]
TASKS=(("depression",fl.FEATS_DEPRESSION,"depression_binary"),("diabetes",fl.FEATS_DIABETES,"diabetes_binary"))
def load(feats,label,seed,mean,std):
    rng=np.random.RandomState(seed); out=[]
    for d in dfs:
        idx=np.arange(len(d)); rng.shuffle(idx); nt=int(len(d)*0.2)
        f=lambda s:(((s[feats].values-mean)/std),s[label].values.astype(float))
        out.append({"train":f(d.iloc[idx[nt:]]),"test":f(d.iloc[idx[:nt]])})
    return out
def pooled_auc_ap(cl,pred):
    P=np.concatenate([sig(c["test"][0]@w+b) for c,(w,b) in zip(cl,pred)]); Y=np.concatenate([c["test"][1] for c in cl])
    return dp.auc_rank(P,Y),float(average_precision_score(Y,P))
def train_local(X,y): return dp.local_train(X,y,np.zeros(X.shape[1]),0.0,STEPS,LR,1.0,0.0,np.inf,None)
def ci(x): x=np.asarray(x,float); return [float(np.nanmean(x)),float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))]

# ---------- (1) pooled baseline
OUT={"pooled":{},"small_site":{}}
for task,feats,lab in TASKS:
    mean,std=fl.federated_feature_stats(dfs,feats); rows=[]
    for s in range(NS):
        cl=load(feats,lab,1000+s,mean,std)
        fw,fb,_=dp.fedavg(cl,LR,None,np.inf,None)
        Xp=np.vstack([c["train"][0] for c in cl]); yp=np.concatenate([c["train"][1] for c in cl])
        pw,pb=train_local(Xp,yp)
        loc=[train_local(*c["train"]) for c in cl]
        fa,fap=pooled_auc_ap(cl,[(fw,fb)]*3); pa,pap=pooled_auc_ap(cl,[(pw,pb)]*3); la,lap=pooled_auc_ap(cl,loc)
        rows.append((fa,pa,la,fap,pap,lap))
    A=np.array(rows)
    OUT["pooled"][task]={"n_splits":NS,"fed_auc":ci(A[:,0]),"pooled_auc":ci(A[:,1]),"local_auc":ci(A[:,2]),
        "fed_minus_pooled_auc":ci(A[:,0]-A[:,1]),"pooled_minus_local_auc":ci(A[:,1]-A[:,2]),
        "fed_ap":ci(A[:,3]),"pooled_ap":ci(A[:,4]),"local_ap":ci(A[:,5]),"fed_minus_pooled_ap":ci(A[:,3]-A[:,4])}
    print("pooled",task,"done",flush=True)

# ---------- (2) small-site experiment
SIZES=[50,100,200,400]
for task,feats,lab in TASKS:
    mean,std=fl.federated_feature_stats(dfs,feats); OUT["small_site"][task]={}
    for k,nm in enumerate(NAMES):
        OUT["small_site"][task][nm]={}
        for n in SIZES+["full"]:
            fa=[];la=[];fap=[];lap=[]
            for r in range(NR):
                cl=load(feats,lab,5000+r,mean,std)
                if n!="full":
                    rng=np.random.RandomState(9000+r); X,y=cl[k]["train"]; i=rng.choice(len(y),n,replace=False); cl[k]["train"]=(X[i],y[i])
                fw,fb,_=dp.fedavg(cl,LR,None,np.inf,None)
                lw,lb=train_local(*cl[k]["train"]); Xt,yt=cl[k]["test"]
                pf=sig(Xt@fw+fb); pl=sig(Xt@lw+lb)
                fa.append(dp.auc_rank(pf,yt)); la.append(dp.auc_rank(pl,yt))
                fap.append(average_precision_score(yt,pf)); lap.append(average_precision_score(yt,pl))
            d=np.array(fa)-np.array(la)
            OUT["small_site"][task][nm][str(n)]={"fed_auc":ci(fa),"loc_auc":ci(la),"fed_minus_local_auc":ci(d),
                "frac_fed_higher":float(np.mean(d>0)),"fed_ap":ci(fap),"loc_ap":ci(lap),"fed_minus_local_ap":ci(np.array(fap)-np.array(lap))}
        print("small",task,nm,"done",flush=True)
json.dump(OUT,open("results/pooled_and_small_site.json","w"),indent=1)
