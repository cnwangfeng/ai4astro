#!/usr/bin/env python3
"""Optional full external-data workflows. Requires an explicitly prepared CSV.
These results are not precomputed in the offline textbook release.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,GroupShuffleSplit,GroupKFold,StratifiedKFold,KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor
from sklearn.dummy import DummyRegressor,DummyClassifier
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,accuracy_score,balanced_accuracy_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]

def run(name):
    filename={'california':'california_housing.csv','airfoil':'airfoil.csv','sonar':'sonar.csv'}[name]
    p=ROOT/'data'/filename
    if not p.exists(): raise FileNotFoundError(f'{p}\nRun scripts/fetch_external_data.py --dataset {name}. No replacement data used.')
    df=pd.read_csv(p);idx=np.arange(len(df));groups=None
    if name=='california':
        features=['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'];target='MedHouseVal'
        tr,te=train_test_split(idx,test_size=.2,random_state=42)
        cv=KFold(5,shuffle=True,random_state=7);scoring='neg_root_mean_squared_error'
        models={'mean':DummyRegressor(),'linear':LinearRegression(),
                'ridge':make_pipeline(StandardScaler(),Ridge(alpha=1.)),
                'forest':RandomForestRegressor(n_estimators=150,min_samples_leaf=5,random_state=42,n_jobs=1),
                'boost':GradientBoostingRegressor(n_estimators=150,max_depth=2,learning_rate=.1,random_state=42)}
        note='Random block-group holdout; not a spatial-extrapolation test. Target unit: $100,000.'
    elif name=='airfoil':
        features=['frequency_hz','angle_deg','chord_m','velocity_ms','displacement_m'];target='sound_db'
        # These are condition-combination proxies, NOT independently supplied physical run identifiers.
        group_cols=['angle_deg','chord_m','velocity_ms','displacement_m']
        groups=pd.factorize(pd.MultiIndex.from_frame(df[group_cols]))[0]
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,groups=groups))
        cv=list(GroupKFold(5).split(df.iloc[tr],groups=groups[tr]));scoring='neg_root_mean_squared_error'
        models={'mean':DummyRegressor(),'linear':make_pipeline(StandardScaler(),Ridge(alpha=1.)),
                'forest':RandomForestRegressor(n_estimators=200,min_samples_leaf=3,random_state=42,n_jobs=1),
                'boost':GradientBoostingRegressor(n_estimators=150,max_depth=2,random_state=42)}
        assert not set(groups[tr])&set(groups[te])
        note='Holdout by identical condition tuples excluding frequency; group IDs are derived proxies, not true experimental run metadata. Target: dB.'
    else:
        features=[f'energy_{i:02d}' for i in range(1,61)];target='target'
        tr,te=train_test_split(idx,test_size=.2,random_state=42,stratify=df[target])
        cv=StratifiedKFold(5,shuffle=True,random_state=7);scoring='balanced_accuracy'
        models={'majority':DummyClassifier(strategy='most_frequent'),
                'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000)),
                'forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=3,random_state=42,n_jobs=1)}
        note='Small-sample random stratified split. No device/run grouping metadata: no claim of unseen-device generalization.'
    missing=[c for c in features+[target] if c not in df]
    if missing: raise ValueError(f'Missing columns: {missing}')
    X=df[features];y=df[target]
    if not np.isfinite(X.to_numpy(float)).all(): raise ValueError('Nonfinite input: design an explicit training-only imputation step.')
    rows=[]
    for model_name,model in models.items():
        s=cross_val_score(model,X.iloc[tr],y.iloc[tr],cv=cv,scoring=scoring)
        if name!='sonar':s=-s
        rows.append({'model':model_name,'CV_mean':float(s.mean()),'fold_std_ddof0':float(s.std())})
    results=pd.DataFrame(rows);selected=results.sort_values('CV_mean',ascending=name!='sonar').iloc[0]['model']
    final=models[selected].fit(X.iloc[tr],y.iloc[tr]);prediction=final.predict(X.iloc[te])
    summary={'dataset':name,'development_n':len(tr),'test_n':len(te),'selected':selected,'scope':note}
    if name=='sonar':
        summary.update(test_accuracy=accuracy_score(y.iloc[te],prediction),test_balanced_accuracy=balanced_accuracy_score(y.iloc[te],prediction),
                       confusion_matrix=confusion_matrix(y.iloc[te],prediction,labels=final.classes_).tolist(),classes=list(final.classes_))
    else: summary.update(test_RMSE=root_mean_squared_error(y.iloc[te],prediction),test_MAE=mean_absolute_error(y.iloc[te],prediction))
    out=ROOT/'results/external'/name;out.mkdir(parents=True,exist_ok=True)
    results.to_csv(out/'development.csv',index=False)
    pd.DataFrame({'row_id':idx,'role':np.where(np.isin(idx,tr),'development','test')}).to_csv(out/'split.csv',index=False)
    pd.DataFrame({'row_id':te,'target':y.iloc[te].to_numpy(),'prediction':prediction}).to_csv(out/'test_predictions.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(results.to_string(index=False));print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True,choices=['california','airfoil','sonar'])
    run(parser.parse_args().dataset)
