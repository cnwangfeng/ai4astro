"""Chapter 4: paired folds, baseline and a deliberately grouped leakage demonstration."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate,GroupKFold,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from common import *

def run():
    X,y,_=development('wine'); splits=list(cv5().split(X,y))
    rows=[]
    models={'majority':DummyClassifier(strategy='most_frequent'),
       'scaled logistic':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000))}
    for name,m in models.items():
        r=cross_validate(m,X,y,cv=splits,scoring=['accuracy','balanced_accuracy'],return_train_score=True)
        for k in range(5):rows.append({'model':name,'fold':k+1,'train_accuracy':r['train_accuracy'][k],
          'validation_accuracy':r['test_accuracy'][k],'balanced_accuracy':r['test_balanced_accuracy'][k]})
    df=table('ch04_paired_folds',rows);show_result(df)
    # Repeat measurements contain object fingerprints unrelated to an independently generated label.
    rng=np.random.default_rng(40);nobj=90;repeats=5
    centers=rng.normal(size=(nobj,8))*3;labels=rng.integers(0,2,nobj)
    groups=np.repeat(np.arange(nobj),repeats)
    features=centers[groups]+rng.normal(scale=.01,size=(nobj*repeats,8));target=labels[groups]
    configurations={'row split':list(StratifiedKFold(5,shuffle=True,random_state=1).split(features,target)),
       'object split':list(GroupKFold(5).split(features,target,groups))}
    leak=[]
    for name,cv in configurations.items():
        r=cross_validate(KNeighborsClassifier(1),features,target,cv=cv,scoring='accuracy')
        for k,v in enumerate(r['test_score']):leak.append({'split':name,'fold':k+1,'accuracy':float(v)})
    df2=table('ch04_group_demo',leak);show_result(df2)
    plt.figure(figsize=(6.4,4))
    for name in configurations:
        s=df2[df2['split']==name];plt.plot(s.fold,s.accuracy,marker='o',label=name)
    plt.ylim(0,1.05);plt.xlabel('折编号');plt.ylabel('验证准确率');plt.legend();figure('v3_ch04_groups')
if __name__=='__main__':run()
