"""Chapter 6: probability, gradient and thresholds with actual boundary cases."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,precision_score,recall_score,log_loss,brier_score_loss
from common import *

def run():
    # Fixed artificial probabilities, not claimed to be fitted model predictions.
    p=np.array([.95,.80,.60,.55,.40,.30,.10,.05]);y=np.array([1,1,0,1,0,1,0,0])
    rows=[]
    for t in [.2,.5,.8]:
        pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        rows.append({'threshold':t,'TP':tp,'FP':fp,'FN':fn,'TN':tn,
        'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0)})
    df=table('ch06_threshold_hand',rows);show_result(df)
    # One simultaneous gradient update, mean binary cross entropy.
    x=np.array([-1.,0.,1.]);target=np.array([0.,0.,1.]);w=0.;b=0.
    prob=expit(w*x+b);dw=np.mean((prob-target)*x);db=np.mean(prob-target)
    wn=w-.3*dw;bn=b-.3*db
    before=log_loss(target,prob);after=log_loss(target,expit(wn*x+bn))
    assert after<before
    dump('ch06_gradient',{'dw':dw,'db':db,'w_new':wn,'b_new':bn,'loss_before':before,'loss_after':after})
    X,y,_=development('wine');splits=list(cv5().split(X,y));rows=[]
    for name,m in [('logistic_C0.1',make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000))),
       ('logistic_C1',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000))),
       ('GaussianNB',GaussianNB())]:
        r=cross_validate(m,X,y,cv=splits,scoring=['accuracy','neg_log_loss'])
        rows.append({'model':name,'accuracy_mean':r['test_accuracy'].mean(),'accuracy_std':r['test_accuracy'].std(),
        'log_loss_mean':-r['test_neg_log_loss'].mean()})
    df=table('ch06_wine_models',rows);show_result(df)
    plt.figure(figsize=(6.4,4));plt.plot(np.arange(len(p)),p,'o',label='prescribed probability')
    for t in [.2,.5,.8]:plt.axhline(t,linestyle='--',label=f'threshold={t}')
    plt.xticks(np.arange(len(p)),[f'{i+1}\ny={v}' for i,v in enumerate(y[:0])]) if False else None
    plt.xlabel('人工样本编号（从0开始）');plt.ylabel('正类概率');plt.legend();figure('v3_ch06_thresholds')
if __name__=='__main__':run()
