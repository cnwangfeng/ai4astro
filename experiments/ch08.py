"""Chapter 8: corrected split enumeration, greedy counterexample and pruning."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,export_text,plot_tree
from sklearn.metrics import accuracy_score,mean_squared_error
from common import *

def impurity(y):
    if not len(y):return 0.
    p=np.bincount(np.asarray(y,dtype=int),minlength=2)/len(y)
    return 1-float(p@p)

def run():
    x=np.array([1.,1.6,2.,2.4,2.8,3.,3.4,3.6,4.2,5.]);y=np.array([0,0,0,0,0,1,0,1,1,1]);rows=[]
    for t in (x[:-1]+x[1:])/2:
        L=y[x<=t];R=y[x>t];weighted=(len(L)*impurity(L)+len(R)*impurity(R))/len(y)
        rows.append({'threshold':t,'left_n':len(L),'right_n':len(R),'weighted_gini':weighted,'gain':impurity(y)-weighted})
    df=table('ch08_split_search',rows);assert np.isclose(df.loc[df.gain.idxmax(),'threshold'],2.9);show_result(df)
    plt.figure(figsize=(6.4,4));plt.plot(df.threshold,df.gain,'o-');plt.xlabel('候选中点阈值');plt.ylabel('Gini下降');figure('v3_ch08_candidates')
    # Counterexample: all comparisons use the same max-depth limit.
    grid=np.array([[i,j] for i in [1,2,3] for j in [1,2,3]])
    labels=np.array([0,0,0,0,0,1,1,1,1]);m=DecisionTreeClassifier(max_depth=2,random_state=42).fit(grid,labels)
    alt=np.where(grid[:,1]<=2.5,np.where(grid[:,0]<=2.5,0,1),np.where(grid[:,0]<=1.5,0,1))
    assert np.sum(m.predict(grid)!=labels)==1 and np.sum(alt!=labels)==0
    dump('ch08_greedy',{'greedy_errors':int(np.sum(m.predict(grid)!=labels)),'alternative_errors':int(np.sum(alt!=labels)),
       'greedy_tree':export_text(m,feature_names=['x1','x2'])})
    plt.figure(figsize=(7,4));plot_tree(m,feature_names=['x1','x2'],class_names=['A','B'],filled=False,rounded=True,fontsize=9);figure('v3_ch08_greedy_tree')
    Xtr,Xva,ytr,yva,ds=development_trainval('iris');rows=[]
    for depth in [1,2,3,4,6,None]:
        t=DecisionTreeClassifier(max_depth=depth,random_state=42).fit(Xtr,ytr)
        rows.append({'max_depth':str(depth),'leaves':t.get_n_leaves(),'training_accuracy':t.score(Xtr,ytr),'validation_accuracy':t.score(Xva,yva)})
    df=table('ch08_iris_depth',rows);show_result(df)
    t=DecisionTreeClassifier(max_depth=2,random_state=42).fit(Xtr,ytr)
    plt.figure(figsize=(9,5));plot_tree(t,feature_names=ds.feature_names,class_names=ds.target_names,filled=False,rounded=True,fontsize=9);figure('v3_ch08_iris_tree')
    sample=Xva.iloc[[0]];path=t.decision_path(sample).indices
    dump('ch08_iris_path',{'row_id':int(sample.index[0]),'values':sample.iloc[0].to_dict(),'true':int(yva.iloc[0]),'predicted':int(t.predict(sample)[0]),'probability':t.predict_proba(sample)[0],'nodes':path,'tree':export_text(t,feature_names=ds.feature_names)})
    # Fixed alpha grid, not selected using held-out test outcomes.
    rows=[]
    for alpha in [0,.005,.01,.02,.04,.08,.16]:
        t=DecisionTreeClassifier(ccp_alpha=alpha,random_state=42).fit(Xtr,ytr)
        rows.append({'alpha':alpha,'leaves':t.get_n_leaves(),'train_accuracy':t.score(Xtr,ytr),'validation_accuracy':t.score(Xva,yva)})
    df=table('ch08_pruning',rows)
    plt.figure(figsize=(6.4,4));plt.plot(df.alpha,df.train_accuracy,'o-',label='training');plt.plot(df.alpha,df.validation_accuracy,'s-',label='validation');plt.xlabel('ccp_alpha');plt.ylabel('准确率');plt.legend();figure('v3_ch08_pruning')
    # Regression demonstration is explicitly synthetic, never labelled housing data.
    xr=np.arange(1,7,dtype=float);yr=np.array([1.,2.,3.,7.,8.,9.]);rows=[]
    for split in xr[:-1]+.5:
        a=yr[xr<=split];b=yr[xr>split];sse=np.sum((a-a.mean())**2)+np.sum((b-b.mean())**2)
        rows.append({'threshold':split,'left_mean':a.mean(),'right_mean':b.mean(),'SSE':sse,'variance_reduction':np.var(yr)-sse/6})
    table('ch08_regression_splits',rows)
    r=DecisionTreeRegressor(max_depth=1,random_state=42).fit(xr[:,None],yr);xx=np.linspace(-1,8,300)
    plt.figure(figsize=(6.4,4));plt.scatter(xr,yr,label='artificial records');plt.plot(xx,r.predict(xx[:,None]),label='one split');plt.xlabel('人工输入 x');plt.ylabel('人工目标 y');plt.legend();figure('v3_ch08_regression')
if __name__=='__main__':run()
