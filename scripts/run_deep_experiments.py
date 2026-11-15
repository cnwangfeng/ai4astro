#!/usr/bin/env python3
"""Run deterministic teaching scripts in fresh processes; no network access needed."""
from pathlib import Path
import argparse,os,sys,subprocess,json,time
R=Path(__file__).resolve().parents[1];out=R/'review_v3';out.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=None)
a=parser.parse_args();names=[f'ch{i:02}' for i in range(1,18)]+['project_ogle']
report=out/'experiment_execution.json'
previous=json.loads(report.read_text()) if report.exists() else {}
if isinstance(previous,list): previous={}
for name in names[a.start:a.stop]:
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 st=time.perf_counter()
 p=subprocess.run([sys.executable,str(R/'experiments'/f'{name}.py')],cwd=R,env=env,capture_output=True,text=True,timeout=240)
 (out/f'{name}_run.log').write_text(p.stdout+'\nSTDERR\n'+p.stderr)
 rec={'exit_code':p.returncode,'elapsed_seconds':round(time.perf_counter()-st,3),'execution':'fresh Python process'}
 previous[name]=rec;report.write_text(json.dumps(previous,indent=2));print(name,rec,flush=True)
 if p.returncode: print(p.stderr,file=sys.stderr);raise SystemExit(p.returncode)
