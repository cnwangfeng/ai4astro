"""Chapter 2: missing values, train-only transforms, and column integrity."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from common import *

def run():
    Xtr,Xva,ytr,yva,ds=development_trainval('wine')
    original=Xtr.copy(); modified=Xtr.copy()
    rng=np.random.default_rng(12)
    positions=rng.choice(len(modified),size=8,replace=False)
    modified.iloc[positions,0]=np.nan
    pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    ztr=pipe.fit_transform(modified);zva=pipe.transform(Xva)
    assert np.isfinite(ztr).all() and np.isfinite(zva).all()
    assert np.allclose(ztr.mean(axis=0),0,atol=1e-12)
    rows=[]
    for j in [0,4,12]:
        rows.append({'feature':Xtr.columns[j], 'training_mean_raw':float(original.iloc[:,j].mean()),
        'training_std_ddof0':float(original.iloc[:,j].std(ddof=0)),
        'training_mean_scaled':float(ztr[:,j].mean()),'validation_mean_scaled':float(zva[:,j].mean())})
    df=table('ch02_scaling',rows);show_result(df)
    table('ch02_missing_positions',[{'training_position':int(i),'original_row':int(Xtr.index[i]),'field':Xtr.columns[0]} for i in positions])
    # A row permutation is harmless only if X and y are permuted together.
    toy=pd.DataFrame({'id':['A','B','C'],'x':[1.,2.,3.]})
    labels=pd.DataFrame({'id':['C','A','B'],'target':[30.,10.,20.]})
    merged=toy.merge(labels,on='id',validate='one_to_one')
    assert np.array_equal(merged.target,[10,20,30])
    table('ch02_safe_join',merged)
if __name__=='__main__':run()
