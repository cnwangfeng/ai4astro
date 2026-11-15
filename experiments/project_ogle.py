"""A real, deliberately limited catalogue excerpt, not a full-survey result."""
from common import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

def run():
    raw = pd.read_csv(DATA/'ogle_lmc_f_excerpt.csv', dtype={'source_id':'string'})
    assert len(raw)==90 and raw.source_id.is_unique
    keep = raw.I_mag.notna() & raw.period_days.notna() & (raw.period_days>0)
    df=raw.loc[keep].copy()
    idx_dev, idx_test = train_test_split(np.arange(len(df)),test_size=.2,random_state=42)
    X=df[['period_days']];y=df.I_mag
    cv=KFold(5,shuffle=True,random_state=7)
    models={'constant':DummyRegressor(), 'linear_period':LinearRegression(),
            'linear_log_period':make_pipeline(FunctionTransformer(np.log10),LinearRegression())}
    rows=[]
    for name,m in models.items():
        values=-cross_val_score(m,X.iloc[idx_dev],y.iloc[idx_dev],cv=cv,scoring='neg_root_mean_squared_error')
        rows.append({'model':name,'cv_rmse':values.mean(),'fold_sd':values.std(ddof=1)})
    scores=table('project_ogle_models',rows)
    name=scores.sort_values('cv_rmse').iloc[0]['model']
    final=models[name].fit(X.iloc[idx_dev],y.iloc[idx_dev])
    prediction=final.predict(X.iloc[idx_test]);err=prediction-y.iloc[idx_test].to_numpy()
    out=df.iloc[idx_test][['source_id','period_days','I_mag']].copy()
    out['prediction_I_mag']=prediction;out['residual_mag']=err
    out.to_csv(RESULTS/'project_ogle_predictions.csv',index=False)
    splits=df[['source_row','source_id']].copy();splits['split']='development';splits.iloc[idx_test,splits.columns.get_loc('split')]='test';splits.to_csv(DATA/'ogle_excerpt_split_v3.csv',index=False)
    linear=final[-1] if hasattr(final,'steps') else final
    summary={'source':'OGLE OCVS LMC fundamental-mode cepF.dat, first 90 records, four source fields',
      'raw_rows':90,'usable_rows':len(df),'excluded_missing_I':int((~keep).sum()),'dev_n':len(idx_dev),'test_n':len(idx_test),
      'selected':name,'test_rmse_mag':root_mean_squared_error(y.iloc[idx_test],prediction),
      'test_mae_mag':mean_absolute_error(y.iloc[idx_test],prediction),'test_bias_mag':float(err.mean()),
      'intercept':float(linear.intercept_),'slope':float(linear.coef_[0]),
      'period_min_days':float(df.period_days.min()),'period_max_days':float(df.period_days.max())}
    dump('project_ogle_summary',summary)
    grid=np.geomspace(df.period_days.min(),df.period_days.max(),160)
    plt.figure(figsize=(6.8,4.4));plt.scatter(np.log10(X.iloc[idx_dev,0]),y.iloc[idx_dev],label='开发记录',s=20)
    plt.scatter(np.log10(X.iloc[idx_test,0]),y.iloc[idx_test],marker='x',label='最终保留记录')
    plt.plot(np.log10(grid),final.predict(pd.DataFrame({'period_days':grid})),label='由开发数据拟合的规则')
    plt.gca().invert_yaxis();plt.xlabel('log10(周期 / 天)');plt.ylabel('I波段强度平均视星等 / mag');plt.legend();figure('v3_project_ogle_fit')
    plt.figure(figsize=(6.8,4.4));plt.scatter(np.log10(out.period_days),out.residual_mag);plt.axhline(0,linestyle='--');plt.xlabel('log10(周期 / 天)');plt.ylabel('预测减目录星等 / mag');figure('v3_project_ogle_residual')
    return summary
if __name__=='__main__':print(run())
