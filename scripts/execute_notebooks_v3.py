#!/usr/bin/env python3
"""Restart-like execution: a separate fresh Jupyter kernel for every notebook.
Use --start/--stop to execute a subset; reports merge by relative filename.
"""
from pathlib import Path
import argparse,time,json,os,traceback,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=None)
args=parser.parse_args();paths=sorted((ROOT/'notebooks').glob('*.ipynb'))[args.start:args.stop]
report=ROOT/'review_v3/notebook_execution.json';records=json.loads(report.read_text()) if report.exists() else {}
for p in paths:
    start=time.perf_counter();rec={'status':'failed','kernel':'fresh python3','code_cells':0}
    try:
        nb=nbformat.read(p,as_version=4);rec['code_cells']=sum(c.cell_type=='code' for c in nb.cells)
        client=NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
        client.execute();nbformat.write(nb,p)
        errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
        if errors:raise RuntimeError(str(errors))
        rec['status']='passed';rec['code_sha256']=hashlib.sha256('\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest()
    except Exception as exc:
        rec['error']=str(exc);rec['traceback']=traceback.format_exc()
    rec['seconds']=round(time.perf_counter()-start,3);records[p.name]=rec
    report.write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(p.name,rec['status'],rec['seconds'],flush=True)
    if rec['status']!='passed':raise SystemExit(1)
