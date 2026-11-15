"""Small, explicit shared utilities. All statistical choices remain in chapter scripts."""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'
FIGURES = ROOT/'figures'
DATA = ROOT/'data'
for p in (RESULTS,FIGURES,DATA): p.mkdir(exist_ok=True)
from matplotlib import font_manager
# TTC files may expose the JP family name while containing all required Chinese glyphs.
_available = {f.name for f in font_manager.fontManager.ttflist}
_cjk = next((name for name in ['Noto Sans CJK SC','Noto Sans CJK JP',
    'Noto Sans SC','Microsoft YaHei','PingFang SC','SimHei'] if name in _available), None)
if _cjk:
    plt.rcParams['font.family'] = ['DejaVu Sans', _cjk]
else:
    import warnings
    warnings.warn('No CJK font found: install a CJK font before regenerating Chinese figures.')
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.size']=10

def frame(name):
    loaders={'iris':load_iris,'wine':load_wine,'digits':load_digits}
    ds=loaders[name](as_frame=True)
    return ds.data.copy(),ds.target.copy(),ds

def development(name):
    """Frozen 80% development split. Test labels are never returned here."""
    X,y,ds=frame(name)
    idx=np.arange(len(y))
    dev,test=train_test_split(idx,test_size=.2,random_state=42,stratify=y)
    path=DATA/f'{name}_split_v3.csv'
    split=pd.DataFrame({'row_id':idx,'role':np.where(np.isin(idx,dev),'development','test')})
    split.to_csv(path,index=False)
    return X.iloc[dev],y.iloc[dev],ds

def development_trainval(name):
    X,y,ds=development(name)
    a,b=train_test_split(np.arange(len(y)),test_size=.25,random_state=7,stratify=y)
    return X.iloc[a],X.iloc[b],y.iloc[a],y.iloc[b],ds

def cv5():
    return StratifiedKFold(5,shuffle=True,random_state=7)

def table(name, rows, columns=None):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    df.to_csv(RESULTS/f'{name}.csv',index=False,float_format='%.10g')
    return df

def tex_table(name, df, headers=None, digits=4):
    """LaTeX fragment, not floating, so its caption belongs to the chapter."""
    out=df.copy()
    if headers: out.columns=headers
    s=out.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.{digits}f}')
    (RESULTS/f'{name}.tex').write_text(s,encoding='utf-8')

def dump(name,obj):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    (RESULTS/f'{name}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert))

def figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES/f'{name}.pdf',bbox_inches='tight')
    plt.savefig(FIGURES/f'{name}.png',dpi=150,bbox_inches='tight')
    plt.close()

def show_result(df):
    print(df.to_string(index=False))

def regression_demo():
    """A declared synthetic response, shared by chapters 5, 8--10 and 16."""
    rng=np.random.default_rng(52)
    x=rng.uniform(-2.5,2.5,300)
    u=rng.normal(size=300)
    clean=1.2+1.5*x+.8*np.sin(3*x)+.25*u
    y=clean+rng.normal(0,.25,300)
    df=pd.DataFrame({'row_id':np.arange(300),'x':x,'u':u,'target':y,'latent_mean':clean})
    df.to_csv(DATA/'synthetic_response.csv',index=False)
    a,b=train_test_split(np.arange(300),test_size=.2,random_state=42)
    tr,va=train_test_split(a,test_size=.25,random_state=7)
    roles=np.full(300,'test',dtype=object);roles[tr]='train';roles[va]='validation'
    table('synthetic_response_split',{'row_id':np.arange(300),'role':roles})
    return df.iloc[tr],df.iloc[va],df.iloc[b]
