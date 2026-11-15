"""Chapter 7: distance metrics, scale and irrelevant dimensions."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist
from common import *

def run():
    X,y,_=development('wine');splits=list(cv5().split(X,y));rows=[]
    for scaled in [False,True]:
        for k in [1,3,5,9,15,25]:
            m=KNeighborsClassifier(k)
            if scaled:m=make_pipeline(StandardScaler(),m)
            s=cross_val_score(m,X,y,cv=splits,scoring='accuracy')
            rows.append({'scaled':scaled,'neighbors':k,'accuracy_mean':s.mean(),'accuracy_std':s.std()})
    df=table('ch07_scale_k',rows);show_result(df)
    plt.figure(figsize=(6.4,4))
    for scaled in [False,True]:
        d=df[df.scaled==scaled];plt.errorbar(d.neighbors,d.accuracy_mean,yerr=d.accuracy_std,marker='o',label=f'scaled={scaled}')
    plt.xlabel('邻居数 k');plt.ylabel('五折准确率（误差棒为折间标准差）');plt.legend();figure('v3_ch07_scale')
    rng=np.random.default_rng(71);rows=[]
    for dim in [2,5,10,30,100,300]:
        P=rng.normal(size=(400,dim));d=pdist(P)
        rows.append({'dimensions':dim,'distance_mean':d.mean(),'distance_std':d.std(),'std_over_mean':d.std()/d.mean()})
    df=table('ch07_dimension',rows)
    plt.figure(figsize=(6.4,4));plt.semilogx(df.dimensions,df.std_over_mean,'o-');plt.xlabel('独立高斯特征维数');plt.ylabel('距离标准差 / 距离均值');figure('v3_ch07_concentration')
    q=np.array([0.,0.]);points=np.array([[1.,0.],[0.,2.],[2.,2.]])
    table('ch07_manual',{'point':['A','B','C'],'distance':np.linalg.norm(points-q,axis=1),'label':[0,1,1]})
if __name__=='__main__':run()
