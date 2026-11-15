"""Chapter 9: averaging, error correlation, finite forest size and OOB."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from common import *

def run():
    Xtr,Xva,ytr,yva,_=development_trainval('wine');rows=[]
    for trees in [10,30,100,300]:
        for seed in range(5):
            m=RandomForestClassifier(n_estimators=trees,max_features='sqrt',random_state=seed,n_jobs=1)
            m.fit(Xtr,ytr);rows.append({'trees':trees,'seed':seed,'validation_accuracy':m.score(Xva,yva)})
    df=table('ch09_forest_seeds',rows);summary=df.groupby('trees').validation_accuracy.agg(['mean','std']).reset_index();table('ch09_forest_summary',summary);show_result(summary)
    plt.figure(figsize=(6.4,4));plt.errorbar(summary.trees,summary['mean'],yerr=summary['std'],marker='o');plt.xlabel('树数量');plt.ylabel('验证准确率（误差棒为种子间标准差）');figure('v3_ch09_tree_count')
    X,y,_=development('wine');splits=list(cv5().split(X,y));rows=[]
    models={'deep tree':DecisionTreeClassifier(random_state=42),
       'bagging':BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,random_state=42,n_jobs=1),
       'forest':RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=42,n_jobs=1)}
    for name,m in models.items():
        s=cross_val_score(m,X,y,cv=splits,scoring='accuracy');rows.append({'model':name,'mean_accuracy':s.mean(),'std_accuracy':s.std()})
    df2=table('ch09_models',rows);show_result(df2)
    oob=RandomForestClassifier(n_estimators=300,oob_score=True,random_state=42,n_jobs=1).fit(Xtr,ytr)
    dump('ch09_oob',{'training_n':len(ytr),'oob_accuracy':oob.oob_score_,'validation_accuracy':oob.score(Xva,yva),'probability_left_out_finite':(1-1/len(ytr))**len(ytr)})
    # Direct Monte Carlo verification of an explicitly idealized covariance model.
    rng=np.random.default_rng(90);rows=[]
    for B in [1,5,20,100]:
        for rho in [0,.2,.8]:
            shared=rng.normal(size=(20000,1));ind=rng.normal(size=(20000,B))
            e=np.sqrt(rho)*shared+np.sqrt(1-rho)*ind
            rows.append({'models':B,'correlation':rho,'theory_variance':rho+(1-rho)/B,'simulated_variance':e.mean(axis=1).var()})
    table('ch09_variance',rows)
    plt.figure(figsize=(6.4,4));B=np.arange(1,101)
    for rho in [0,.2,.8]:plt.plot(B,rho+(1-rho)/B,label=f'rho={rho}')
    plt.xlabel('模型数 B');plt.ylabel('平均预测的理论方差（单模型方差=1）');plt.legend();figure('v3_ch09_correlation')
if __name__=='__main__':run()
