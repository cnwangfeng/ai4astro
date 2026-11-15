#!/usr/bin/env python3
"""Download optional real datasets, or verify the supplied OGLE excerpt.

Network access is required. Failures are explicit; no substitute dataset is used.
Downloaded source bytes and checksums are recorded. Existing source results are
not silently treated as results for a different version.
"""
from __future__ import annotations
import argparse, hashlib, io, json, sys, urllib.request, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; RAW=DATA/'external_raw'
OGLE_URL='https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/cepF.dat'
README_URL='https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/README'

def download(url: str, dest: Path, timeout: float) -> bytes:
    req=urllib.request.Request(url,headers={'User-Agent':'ML-Physics-Textbook/3.0 (educational data retrieval)'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            payload=response.read()
    except Exception as exc:
        raise RuntimeError(f'Download failed: {url}\n{exc}\nNo replacement data were used.') from exc
    if not payload: raise RuntimeError(f'Empty response: {url}')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    provenance={'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    dest.with_suffix(dest.suffix+'.source.json').write_text(json.dumps(provenance,indent=2))
    return payload

def parse_ogle(payload: bytes) -> pd.DataFrame:
    records=[]
    for line in payload.decode('utf-8').splitlines():
        fields=line.split()
        if not fields or not fields[0].startswith('OGLE-LMC-CEP-'): continue
        if len(fields)<4: raise ValueError('Unexpected OGLE row: '+line)
        val=lambda x: np.nan if x=='-' else float(x)
        records.append({'source_row':len(records),'source_id':fields[0],
                        'I_mag':val(fields[1]),'V_mag':val(fields[2]),'period_days':val(fields[3])})
    df=pd.DataFrame(records)
    if df.empty or not df.source_id.is_unique: raise ValueError('Empty/duplicate OGLE IDs')
    if not (df.period_days.dropna()>0).all(): raise ValueError('Nonpositive period')
    return df

def fetch(dataset: str, timeout: float) -> Path:
    DATA.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    if dataset in {'ogle-full','ogle-verify'}:
        payload=download(OGLE_URL,RAW/'cepF.dat',timeout)
        download(README_URL,RAW/'OGLE_README.txt',timeout)
        df=parse_ogle(payload)
        if dataset=='ogle-verify':
            local=pd.read_csv(DATA/'ogle_lmc_f_excerpt.csv',dtype={'source_id':'string'})
            selected=df.iloc[:len(local)].copy()
            if selected.source_id.tolist()!=local.source_id.tolist():
                raise ValueError('Source IDs or ordering changed. Do not overwrite the fixed excerpt.')
            for c in ['I_mag','V_mag','period_days']:
                if not np.allclose(selected[c],local[c],equal_nan=True,rtol=0,atol=5e-8):
                    raise ValueError(f'Field {c} differs from the fixed excerpt. Inspect source version.')
            p=ROOT/'review_v3/ogle_online_verification.json';p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps({'status':'passed','rows':len(local),'source':OGLE_URL,
               'utc':datetime.now(timezone.utc).isoformat()},indent=2))
            return p
        out=DATA/'ogle_lmc_f_full.csv';df.to_csv(out,index=False);return out
    if dataset=='california':
        from sklearn.datasets import fetch_california_housing
        # sklearn also verifies its distributed download checksum.
        try: ds=fetch_california_housing(as_frame=True,data_home=str(RAW/'sklearn_cache'))
        except Exception as exc: raise RuntimeError('California Housing download failed; no fallback used.') from exc
        df=ds.frame.copy();df.insert(0,'record_id',np.arange(len(df)))
        if len(df)!=20640: raise ValueError(f'Unexpected Housing rows: {len(df)}')
        out=DATA/'california_housing.csv';df.to_csv(out,index=False)
        out.with_suffix('.source.json').write_text(json.dumps({'source':'sklearn.datasets.fetch_california_housing',
            'utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))
        return out
    configs={
      'airfoil':('https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip',
         'airfoil_self_noise.dat',['frequency_hz','angle_deg','chord_m','velocity_ms','displacement_m','sound_db'],r'\s+',1503),
      'sonar':('https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip',
         'sonar.all-data',[f'energy_{i:02d}' for i in range(1,61)]+['target'],',',208)}
    url,suffix,names,sep,n=configs[dataset]
    payload=download(url,RAW/f'{dataset}.zip',timeout)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        matches=[p for p in z.namelist() if p.endswith(suffix)]
        if len(matches)!=1: raise ValueError(f'Unexpected archive members for {suffix}: {matches}')
        df=pd.read_csv(io.BytesIO(z.read(matches[0])),header=None,sep=sep,names=names)
    if df.shape!=(n,len(names)): raise ValueError(f'Unexpected {dataset} shape {df.shape}')
    if dataset=='sonar' and set(df.target)!={'M','R'}: raise ValueError('Unexpected sonar target codes')
    df.insert(0,'record_id',np.arange(len(df)));out=DATA/f'{dataset}.csv';df.to_csv(out,index=False)
    return out

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['california','airfoil','sonar','ogle-full','ogle-verify'])
    parser.add_argument('--timeout',type=float,default=45)
    args=parser.parse_args()
    try: print(fetch(args.dataset,args.timeout)); return 0
    except Exception as exc: print(str(exc),file=sys.stderr); return 1
if __name__=='__main__': raise SystemExit(main())
