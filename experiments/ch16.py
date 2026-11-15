"""Chapter 16: residual groups, correlated features and permutation importance."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from common import *

def run():
    tr,va,_=regression_demo();cols=['x','u']
    m=RandomForestRegressor(n_estimators=150,min_samples_leaf=3,random_state=42,n_jobs=1).fit(tr[cols],tr.target)
    pred=m.predict(va[cols]);errors=pred-va.target.to_numpy();rows=[]
    for lower,upper in [(-2.5,-1),(-1,1),(1,2.5)]:
        mask=(va.x>=lower)&(va.x<upper)
        rows.append({'group':f'[{lower},{upper})','count':int(mask.sum()),'mean_error':float(errors[mask].mean()),'RMSE':float(np.sqrt(np.mean(errors[mask]**2)))})
    df=table('ch16_group_errors',rows);show_result(df)
    table('ch16_residual_records',{'row_id':va.row_id.to_numpy(),'x':va.x.to_numpy(),'target':va.target.to_numpy(),'prediction':pred,'prediction_minus_target':errors})
    plt.figure(figsize=(6.4,4));plt.scatter(va.x,errors,s=20);plt.axhline(0,linestyle='--');plt.xlabel('人工输入 x');plt.ylabel('预测减目标（人工单位）');figure('v3_ch16_residuals')
    r=permutation_importance(m,va[cols],va.target,n_repeats=15,random_state=17,scoring='neg_root_mean_squared_error')
    table('ch16_importance',{'feature':cols,'mean_RMSE_increase':r.importances_mean,'repeat_std':r.importances_std})
    # Bootstrap uncertainty for the *fixed* predictions, not training-set uncertainty.
    rng=np.random.default_rng(160);boot=[]
    for b in range(1000):
        ids=rng.integers(0,len(errors),len(errors));boot.append(np.sqrt(np.mean(errors[ids]**2)))
    dump('ch16_fixed_model_bootstrap',{'RMSE':np.sqrt(np.mean(errors**2)),'percentile_2.5':np.quantile(boot,.025),'percentile_97.5':np.quantile(boot,.975),'scope':'fixed predictions, IID row resampling of synthetic validation sample'})
if __name__=='__main__':run()
