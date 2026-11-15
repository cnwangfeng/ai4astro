"""Chapter 15: scores are rankings; thresholds require evaluation and a reference set."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score,recall_score,average_precision_score
from common import *

def run():
    rng=np.random.default_rng(150)
    reference=rng.normal(size=(400,2))
    normal=rng.normal(size=(200,2));anomaly=rng.uniform(4,7,size=(20,2));X=np.vstack([normal,anomaly]);y=np.r_[np.zeros(200,int),np.ones(20,int)]
    m=IsolationForest(n_estimators=200,contamination='auto',random_state=42).fit(reference)
    refscore=-m.score_samples(reference);score=-m.score_samples(X);rows=[]
    for q in [.90,.95,.99]:
        threshold=np.quantile(refscore,q);p=score>threshold
        rows.append({'reference_quantile':q,'threshold':threshold,'candidates':int(p.sum()),'precision':precision_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'normal_false_positive_rate':float(p[:200].mean())})
    df=table('ch15_thresholds',rows);show_result(df)
    table('ch15_candidates',{'row_id':np.arange(len(y)),'x1':X[:,0],'x2':X[:,1],'injected_label':y,'anomaly_score':score})
    np.savez(DATA/'synthetic_anomaly_reference.npz',reference=reference,X=X,y=y)
    dump('ch15_ranking',{'average_precision':average_precision_score(y,score),'n_reference':len(reference),'n_query':len(y)})
    plt.figure(figsize=(6.4,4));plt.hist(score[y==0],bins=20,alpha=.6,label='normal queries');plt.hist(score[y==1],bins=15,alpha=.6,label='injected queries');plt.xlabel('负 score_samples：越大越异常');plt.ylabel('记录数');plt.legend();figure('v3_ch15_scores')
    # A local-density example has different meanings from a global score.
    dense=rng.normal(0,.15,(120,2));sparse=rng.normal([3,0],[.7,.7],(120,2));P=np.vstack([dense,sparse,[[.8,0]]])
    lof=LocalOutlierFactor(n_neighbors=20).fit(P);s=-lof.negative_outlier_factor_
    table('ch15_local',{'row_id':np.arange(len(P)),'x1':P[:,0],'x2':P[:,1],'LOF':s})
    plt.figure(figsize=(6.4,4));plt.scatter(P[:,0],P[:,1],s=15+20*np.clip(s-1,0,5));plt.xlabel('人工坐标 x1');plt.ylabel('人工坐标 x2');figure('v3_ch15_local_density')
if __name__=='__main__':run()
