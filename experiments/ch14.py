"""Chapter 14: K-means updates, non-convex shapes and stability."""
import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score,silhouette_score
from common import *

def run():
    x=np.array([0.,1.,4.,5.,9.,10.]);centers=np.array([0.,4.]);records=[]
    for step in range(5):
        labels=np.argmin(abs(x[:,None]-centers[None,:]),axis=1)
        new=np.array([x[labels==k].mean() for k in range(2)])
        sse=np.sum((x-new[labels])**2)
        records.append({'iteration':step+1,'center0_before':centers[0],'center1_before':centers[1],'center0_after':new[0],'center1_after':new[1],'within_SSE_after':sse,'labels':' '.join(map(str,labels))})
        centers=new
    df=table('ch14_lloyd',records);show_result(df)
    X,truth=make_moons(n_samples=350,noise=.07,random_state=42);rows=[]
    models={'K-means':KMeans(2,n_init=10,random_state=42),'DBSCAN':DBSCAN(eps=.22,min_samples=5),'single-link':AgglomerativeClustering(2,linkage='single')}
    for name,m in models.items():
        labels=m.fit_predict(X);non=labels>=0;clusters=len(set(labels)-{-1})
        sil=float(silhouette_score(X[non],labels[non])) if clusters>=2 else np.nan
        rows.append({'method':name,'clusters':clusters,'noise_count':int((~non).sum()),'ARI_to_generator':adjusted_rand_score(truth,labels),'silhouette_nonnoise':sil})
        plt.figure(figsize=(5.6,4.3))
        for lab in sorted(set(labels)):
            mask=labels==lab;plt.scatter(X[mask,0],X[mask,1],s=13,marker='x' if lab==-1 else 'o',label=f'group {lab}')
        plt.xlabel('人工坐标 x1');plt.ylabel('人工坐标 x2');plt.legend();figure('v3_ch14_'+name.replace('-','_'))
    df=table('ch14_shapes',rows);show_result(df)
    rows=[]
    reference=DBSCAN(eps=.22,min_samples=5).fit_predict(X)
    for eps in [.1,.14,.18,.22,.3,.5]:
        lab=DBSCAN(eps=eps,min_samples=5).fit_predict(X)
        rows.append({'eps':eps,'clusters':len(set(lab)-{-1}),'noise':int((lab==-1).sum()),'ARI_to_reference':adjusted_rand_score(reference,lab)})
    table('ch14_eps',rows)
if __name__=='__main__':run()
