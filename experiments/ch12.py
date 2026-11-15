"""Chapter 12: an explicit XOR network, backprop check and training diagnostics."""
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss,accuracy_score
from sklearn.exceptions import ConvergenceWarning
from common import *

def run():
    X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float);y=np.array([0,1,1,0])
    h1=np.maximum(X[:,0]-X[:,1],0);h2=np.maximum(X[:,1]-X[:,0],0)
    out=h1+h2;assert np.array_equal(out,y)
    table('ch12_xor',{'x1':X[:,0],'x2':X[:,1],'h1':h1,'h2':h2,'output':out,'target':y})
    def loss(t):
        w1,b1,w2,b2=t;pred=w2*max(0,w1*2+b1)+b2
        return .5*(pred-1)**2
    t=np.array([.5,0.,3.,0.]);g=np.array([12.,6.,2.,2.]);eps=1e-6
    gn=np.array([(loss(t+eps*np.eye(4)[j])-loss(t-eps*np.eye(4)[j]))/(2*eps) for j in range(4)])
    assert np.allclose(g,gn,rtol=1e-6)
    table('ch12_backprop',{'parameter':['w1','b1','w2','b2'],'analytic':g,'finite_difference':gn,'updated':t-.01*g})
    dump('ch12_update',{'loss_before':loss(t),'loss_after':loss(t-.01*g)})
    Xtr,Xva,ytr,yva,_=development_trainval('wine');rows=[];warning_log=[]
    for width in [4,12,32]:
        for seed in [0,1,2,3,4]:
            m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(width,),solver='lbfgs',alpha=.1,max_iter=1500,random_state=seed))
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always');m.fit(Xtr,ytr)
            warning_log.extend([{'width':width,'seed':seed,'message':str(w.message)} for w in ws])
            rows.append({'hidden_width':width,'seed':seed,'train_accuracy':m.score(Xtr,ytr),'validation_accuracy':m.score(Xva,yva),'validation_log_loss':log_loss(yva,m.predict_proba(Xva)),'iterations':m[-1].n_iter_})
    df=table('ch12_initializations',rows);summary=df.groupby('hidden_width').validation_accuracy.agg(['mean','std']).reset_index();table('ch12_initialization_summary',summary);show_result(summary);dump('ch12_warnings',warning_log)
    sc=StandardScaler().fit(Xtr);zt=sc.transform(Xtr);zv=sc.transform(Xva)
    m=MLPClassifier(hidden_layer_sizes=(12,),solver='sgd',learning_rate_init=.02,batch_size=20,alpha=.1,random_state=3,shuffle=True)
    records=[]
    for epoch in range(1,161):
        m.partial_fit(zt,ytr,classes=np.unique(ytr))
        records.append({'epoch':epoch,'train_log_loss':log_loss(ytr,m.predict_proba(zt)),'validation_log_loss':log_loss(yva,m.predict_proba(zv))})
    df=table('ch12_training_trace',records)
    plt.figure(figsize=(6.4,4));plt.plot(df.epoch,df.train_log_loss,label='training');plt.plot(df.epoch,df.validation_log_loss,label='validation');plt.xlabel('训练轮数 epoch');plt.ylabel('数据交叉熵（不含正则项）');plt.legend();figure('v3_ch12_training')
if __name__=='__main__':run()
