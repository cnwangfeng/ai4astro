"""Reproducible numeric-table baseline for externally prepared science CSVs.

Example (bundled Wine):
python scripts/run_science_csv.py --csv data/wine.csv \
  --target target --features 'alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280/od315_of_diluted_wines,proline' \
  --task classification --out results/wine_project

Actual SDSS/OGLE/Gaia/Airfoil/Sonar projects require verified downloaded data,
a data dictionary, science-specific selection, and an appropriate group column.
This program does not download or fabricate any observational data.
"""
from __future__ import annotations
import argparse, json, platform, pathlib, warnings
import numpy as np, pandas as pd, sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, GroupKFold,
                                     StratifiedKFold, KFold, cross_validate)
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             mean_absolute_error, root_mean_squared_error)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',required=True)
    p.add_argument('--target',required=True)
    p.add_argument('--features',required=True,help='comma-separated explicit numeric input columns')
    p.add_argument('--task',required=True,choices=['classification','regression'])
    p.add_argument('--group-col',default=None)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    cols=[x.strip() for x in a.features.split(',') if x.strip()]
    if a.target in cols or (a.group_col and a.group_col in cols):
        raise ValueError('Target or grouping field cannot be a predictor.')
    if len(cols)<1: raise ValueError('At least one feature must be named.')
    d=pd.read_csv(a.csv)
    required=cols+[a.target]+([a.group_col] if a.group_col else [])
    missing=set(required)-set(d.columns)
    if missing:raise KeyError(f'Columns missing from CSV: {sorted(missing)}')
    d=d.dropna(subset=[a.target]+([a.group_col] if a.group_col else [])).copy()
    X=d.loc[:,cols].copy()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise TypeError(f'Non-numeric feature {col}; encode categories in an audited pipeline.')
    X=X.replace([np.inf,-np.inf],np.nan)
    y=d[a.target].copy()
    groups=d[a.group_col].to_numpy() if a.group_col else None
    if a.task=='classification':
        if y.nunique()<2:raise ValueError('Need at least 2 target classes.')
        if y.value_counts().min()<2:raise ValueError('Classes too rare for holdout stratification.')
    if groups is None:
        ids_tr,ids_te=train_test_split(np.arange(len(d)),test_size=.2,random_state=a.seed,
             stratify=y if a.task=='classification' else None)
        n_splits=min(5,len(ids_tr))
        if a.task=='classification':
            n_splits=min(n_splits,int(y.iloc[ids_tr].value_counts().min()))
        if n_splits<2:raise ValueError('Not enough samples per class for CV.')
        cv=(StratifiedKFold(n_splits,shuffle=True,random_state=a.seed+1)
            if a.task=='classification' else KFold(n_splits,shuffle=True,random_state=a.seed+1))
        fit_groups=None
    else:
        splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=a.seed)
        ids_tr,ids_te=next(splitter.split(X,y,groups=groups))
        n_groups=len(np.unique(groups[ids_tr]))
        if n_groups<2:raise ValueError('At least two development groups required.')
        cv=GroupKFold(n_splits=min(5,n_groups))
        fit_groups=groups[ids_tr]
    Xtr,Xte=X.iloc[ids_tr],X.iloc[ids_te]
    ytr,yte=y.iloc[ids_tr],y.iloc[ids_te]
    if a.task=='classification':
        models={
            'majority_baseline':DummyClassifier(strategy='most_frequent'),
            'logistic':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                                   LogisticRegression(max_iter=3000)),
            'random_forest':make_pipeline(SimpleImputer(strategy='median'),
                              RandomForestClassifier(n_estimators=160,min_samples_leaf=2,
                                                     random_state=a.seed,n_jobs=1)),
        }
        score='balanced_accuracy'
    else:
        models={
            'mean_baseline':DummyRegressor(strategy='mean'),
            'ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10.0)),
            'random_forest':make_pipeline(SimpleImputer(strategy='median'),
                             RandomForestRegressor(n_estimators=160,min_samples_leaf=3,
                                                   random_state=a.seed,n_jobs=1)),
        }
        score='neg_mean_absolute_error'
    rows=[]
    for name,model in models.items():
        kwargs={'cv':cv,'scoring':score,'n_jobs':1,'error_score':'raise'}
        if fit_groups is not None:kwargs['groups']=fit_groups
        with sklearn.config_context(enable_metadata_routing=False):
            result=cross_validate(model,Xtr,ytr,**kwargs)['test_score']
        value=result if a.task=='classification' else -result
        rows.append(dict(model=name,mean=float(value.mean()),std=float(value.std()),
                         n_folds=len(value)))
    ranking=pd.DataFrame(rows).sort_values('mean',ascending=(a.task=='regression'))
    chosen=ranking.iloc[0]['model']
    selected=models[chosen].fit(Xtr,ytr)
    pred=selected.predict(Xte)
    if a.task=='classification':
        final={'balanced_accuracy':float(balanced_accuracy_score(yte,pred)),
               'classification_report':classification_report(yte,pred,output_dict=True,zero_division=0)}
    else:
        final={'MAE':float(mean_absolute_error(yte,pred)),
               'RMSE':float(root_mean_squared_error(yte,pred))}
    dest=pathlib.Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    ranking.to_csv(dest/'cv_comparison.csv',index=False)
    pd.DataFrame({'original_row':d.index.to_numpy()[ids_te],
                  'target':yte.to_numpy(),'predicted':pred}).to_csv(dest/'final_predictions.csv',index=False)
    (dest/'run_record.json').write_text(json.dumps({
        'dataset':str(pathlib.Path(a.csv).resolve()),'target':a.target,'features':cols,
        'group_col':a.group_col,'seed':a.seed,'task':a.task,'sample_count':len(d),
        'development_count':len(ids_tr),'holdout_count':len(ids_te),
        'scoring':score,'selected_using_development_cv':chosen,
        'final_holdout_metrics':final,'sklearn_version':sklearn.__version__,
        'python_version':platform.python_version()},indent=2,ensure_ascii=False),encoding='utf-8')
    print('Selected from development CV:',chosen)
    print(ranking.to_string(index=False))
    print('Final holdout:',{k:v for k,v in final.items() if k!='classification_report'})
    print('Results:',dest)
if __name__=='__main__':main()
