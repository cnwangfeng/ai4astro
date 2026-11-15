"""Chapter 10: forward-stagewise residual fitting and validation-selected iteration."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from common import *

def run():
    x=np.arange(1,5,dtype=float)[:,None];y=np.array([0.,1.,2.,4.]);pred=np.full(4,y.mean());rows=[]
    rows.append({'stage':0,'prediction':pred.tolist(),'MSE':np.mean((y-pred)**2)})
    detailed=[]
    for stage in [1,2]:
        residual=y-pred;tree=DecisionTreeRegressor(max_depth=1,random_state=42).fit(x,residual)
        correction=tree.predict(x);new=pred+.5*correction
        for i in range(4):detailed.append({'stage':stage,'x':x[i,0],'target':y[i],'old_prediction':pred[i],'residual':residual[i],'tree_prediction':correction[i],'new_prediction':new[i]})
        pred=new;rows.append({'stage':stage,'prediction':pred.tolist(),'MSE':np.mean((y-pred)**2)})
    dump('ch10_two_steps',rows);df=table('ch10_hand',detailed);show_result(df)
    tr,va,_=regression_demo();cols=['x','u'];allrows=[]
    for lr in [.03,.1,.3]:
        m=GradientBoostingRegressor(n_estimators=220,learning_rate=lr,max_depth=2,random_state=42).fit(tr[cols],tr.target)
        for k,(a,b) in enumerate(zip(m.staged_predict(tr[cols]),m.staged_predict(va[cols])),1):
            allrows.append({'learning_rate':lr,'stage':k,'train_RMSE':np.sqrt(mean_squared_error(tr.target,a)),
            'validation_RMSE':np.sqrt(mean_squared_error(va.target,b))})
    df=table('ch10_learning_curves',allrows);summary=df.loc[df.groupby('learning_rate').validation_RMSE.idxmin()];table('ch10_selected_stages',summary);show_result(summary)
    plt.figure(figsize=(6.4,4))
    for lr in [.03,.1,.3]:
        d=df[df.learning_rate==lr];plt.plot(d.stage,d.validation_RMSE,label=f'eta={lr}')
    plt.xlabel('提升轮数');plt.ylabel('验证RMSE（人工单位）');plt.legend();figure('v3_ch10_stages')
    # These are development comparisons; final test data are intentionally not touched.
    rows=[]
    for name,m in [('tree',DecisionTreeRegressor(max_depth=4,random_state=42)),('forest',RandomForestRegressor(n_estimators=200,min_samples_leaf=3,random_state=42,n_jobs=1)),('boosting',GradientBoostingRegressor(n_estimators=120,max_depth=2,learning_rate=.1,random_state=42))]:
        m.fit(tr[cols],tr.target);rows.append({'model':name,'train_RMSE':np.sqrt(mean_squared_error(tr.target,m.predict(tr[cols]))),'validation_RMSE':np.sqrt(mean_squared_error(va.target,m.predict(va[cols])))})
    table('ch10_comparison',rows)
if __name__=='__main__':run()
