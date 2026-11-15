"""Chapter 17: an entirely reproducible, explicitly simulated physical calibration project.
No output is represented as a result from a telescope or laboratory observation.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit,GroupKFold,cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from common import *

def run():
    rng=np.random.default_rng(170)
    # 90 measurement sessions. Each session shares a zero-point offset and instrument temperature.
    nrun=90;perrun=8;run_id=np.repeat(np.arange(nrun),perrun)
    temp_run=rng.uniform(15,35,nrun);offset_run=rng.normal(0,.12,nrun)
    x=rng.uniform(0,10,nrun*perrun);T=temp_run[run_id]
    truth=.5+1.8*x+.06*x*x+.02*(T-25)*x
    measured=truth+offset_run[run_id]+rng.normal(0,.08,len(x))
    df=pd.DataFrame({'record_id':np.arange(len(x)),'run_id':run_id,'input_V':x,'temperature_C':T,'output_V':measured})
    df.to_csv(DATA/'simulated_calibration_project.csv',index=False)
    dev,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,groups=run_id))
    assert not (set(run_id[dev])&set(run_id[test]))
    roles=np.full(len(df),'test',dtype=object);roles[dev]='development';table('ch17_splits',{'record_id':df.record_id,'run_id':run_id,'role':roles})
    X=df[['input_V','temperature_C']];y=df.output_V;cv=list(GroupKFold(5).split(X.iloc[dev],y.iloc[dev],run_id[dev]))
    models={'mean':DummyRegressor(strategy='mean'),'linear':LinearRegression(),
      'quadratic_ridge':make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=.01)),
      'forest':RandomForestRegressor(n_estimators=200,min_samples_leaf=3,random_state=42,n_jobs=1)}
    rows=[]
    for name,m in models.items():
        s=-cross_val_score(m,X.iloc[dev],y.iloc[dev],cv=cv,scoring='neg_root_mean_squared_error')
        rows.append({'model':name,'CV_RMSE_V':s.mean(),'fold_std_V':s.std()})
    scores=table('ch17_development',rows);show_result(scores)
    selected=scores.sort_values('CV_RMSE_V').iloc[0]['model'];m=models[selected].fit(X.iloc[dev],y.iloc[dev]);pred=m.predict(X.iloc[test])
    errors=pred-y.iloc[test].to_numpy();by=[]
    for group in np.unique(run_id[test]):
        e=errors[run_id[test]==group];by.append({'run_id':int(group),'RMSE_V':np.sqrt(np.mean(e*e)),'mean_error_V':e.mean()})
    table('ch17_test_by_run',by);table('ch17_test_predictions',{'record_id':df.record_id.iloc[test].to_numpy(),'run_id':run_id[test],'target_V':y.iloc[test].to_numpy(),'prediction_V':pred,'residual_V':errors})
    out={'data_status':'synthetic physical teaching project','selected_model':selected,'development_records':len(dev),'test_records':len(test),'development_groups':len(set(run_id[dev])),'test_groups':len(set(run_id[test])),'test_RMSE_V':np.sqrt(np.mean(errors**2)),'test_MAE_V':np.mean(abs(errors))}
    dump('ch17_final',out);print(out)
    plt.figure(figsize=(6.4,4));plt.scatter(y.iloc[test],pred,s=14);a,b=float(y.min()),float(y.max());plt.plot([a,b],[a,b],linestyle='--');plt.xlabel('模拟测量输出（V）');plt.ylabel('最终模型预测（V）');figure('v3_ch17_predictions')
if __name__=='__main__':run()
