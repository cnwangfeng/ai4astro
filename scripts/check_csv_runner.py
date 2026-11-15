"""Smoke tests for the shared CSV runner; synthetic groups are not science data."""
from pathlib import Path
import tempfile,subprocess,sys,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
results=[]
def run(name,args,should_pass=True):
    p=subprocess.run([sys.executable,str(R/'scripts/run_science_csv.py'),*args],cwd=R,capture_output=True,text=True)
    ok=(p.returncode==0) if should_pass else (p.returncode!=0)
    results.append({'case':name,'status':'PASS' if ok else 'FAIL','expected_success':should_pass,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    if not ok:raise RuntimeError(p.stdout+'\n'+p.stderr)
with tempfile.TemporaryDirectory(prefix='ml_csv_check_') as tmp:
    t=Path(tmp)
    run('bundled wine classification',['--csv','data/wine.csv','--target','target','--features','alcohol,malic_acid,ash','--task','classification','--out',str(t/'wine')])
    run('named auxiliary regression',['--csv','data/diabetes.csv','--target','target','--features','age,bmi,bp','--task','regression','--out',str(t/'reg')])
    rng=np.random.default_rng(42);x=rng.normal(size=(120,2));labels=(x[:,0]+.5*x[:,1]>0).astype(int)
    data=pd.DataFrame({'x1':x[:,0],'x2':x[:,1],'target':labels,'synthetic_group':np.repeat(np.arange(12),10)})
    data.to_csv(t/'groups.csv',index=False)
    run('artificial grouping API test',['--csv',str(t/'groups.csv'),'--target','target','--features','x1,x2','--task','classification','--group-col','synthetic_group','--out',str(t/'group')])
    run('reject target as predictor',['--csv','data/wine.csv','--target','target','--features','alcohol,target','--task','classification','--out',str(t/'bad')],False)
(R/'review'/'csv_runner_checks.json').write_text(json.dumps({'scope':'API smoke tests only; no external science data validation','results':results},ensure_ascii=False,indent=2))
print(len(results),'CSV runner checks passed.')
