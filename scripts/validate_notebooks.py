"""Execute each notebook in a fresh Jupyter kernel, without hidden prior state.
Usage: python scripts/validate_notebooks.py --jobs 2
Outputs test records and executed notebook copies under review/.
"""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
import argparse, json, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
def run(path):
    start=time.monotonic(); result={'notebook':path.name}
    nb=nbformat.read(path,4)
    try:
        client=NotebookClient(nb,timeout=180,kernel_name='python3',allow_errors=False,
                              resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        target=ROOT/'review'/'executed_notebooks'/path.name
        target.parent.mkdir(parents=True,exist_ok=True)
        nbformat.write(nb,target)
        warnings=[]
        for cell in nb.cells:
            for out in cell.get('outputs',[]):
                if out.get('output_type')=='stream' and out.get('name')=='stderr':
                    warnings.append(out.get('text',''))
        result.update(status='PASS',code_cells=sum(c.cell_type=='code' for c in nb.cells),
                      warnings=warnings)
    except Exception as exc:
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc())
    result['seconds']=round(time.monotonic()-start,3)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--pattern',default='ch*_*.ipynb');args=parser.parse_args()
    paths=sorted((ROOT/'notebooks').glob(args.pattern));results=[]
    if not paths:raise SystemExit('No matching notebooks')
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        fs={executor.submit(run,p):p for p in paths}
        for future in as_completed(fs):
            r=future.result();results.append(r);print(r['status'],r['notebook'],r['seconds'],flush=True)
    results.sort(key=lambda r:r['notebook'])
    report={'method':'nbclient; one fresh Jupyter Python kernel per notebook',
            'scope':'offline default paths, not optional network datasets or classroom trial',
            'passed':sum(r['status']=='PASS' for r in results),'total':len(results),'results':results}
    (ROOT/'review'/'notebook_execution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if any(r['status']!='PASS' for r in results):raise SystemExit(1)
if __name__=='__main__':main()
