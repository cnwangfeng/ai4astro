#!/usr/bin/env python3
"""Check the delivered source graph, assets, counts, and recorded execution identity.
This is a structural/technical check, not a replacement for academic peer review.
"""
from pathlib import Path
import json,re,hashlib,sys,collections
import nbformat
ROOT=Path(__file__).resolve().parents[1]
errors=[]; visited=[]
def expand(path):
    if not path.exists():errors.append(f'Missing TeX: {path}');return ''
    if path in visited:errors.append(f'Repeated input: {path.relative_to(ROOT)}');return ''
    visited.append(path);text=path.read_text(encoding='utf-8')
    def sub(m):
        q=ROOT/m.group(1)
        if not q.suffix:q=q.with_suffix('.tex')
        return expand(q)
    return re.sub(r'\\input\{([^}]+)\}',sub,text)
text=expand(ROOT/'main.tex')
labels=re.findall(r'\\label\{([^}]+)\}',text)
dup=[k for k,v in collections.Counter(labels).items() if v>1]
if dup:errors.append('Duplicate labels: '+str(dup))
refs=re.findall(r'\\(?:ref|eqref|pageref|cref|Cref)\{([^}]+)\}',text)
missing=sorted(set(refs)-set(labels))
if missing:errors.append('Missing references: '+str(missing))
assets=[]
for match in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
    if '#' in match:continue
    candidates=[ROOT/match]+[ROOT/(match+ext) for ext in ['.pdf','.png','.jpg']]
    if not any(p.exists() for p in candidates):errors.append('Missing graphic: '+match)
    assets.append(match)
for match in re.findall(r'\\fig(?:\[[^]]*\])?\{([^}]+)\}',text):
    if not (ROOT/'figures'/match).exists():errors.append('Missing fig: '+match)
    assets.append('figures/'+match)
chapters=[]
for p in sorted((ROOT/'chapters').glob('ch??.tex')):
    t=p.read_text();plain=re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}','',t,flags=re.S)
    nsec=len(re.findall(r'\\section\{',t));dmatch=re.search(r'\\begin\{enumerate\}\[label=D\\arabic\*\.\](.*?)\\end\{enumerate\}',t,re.S)
    nd=len(re.findall(r'\\item\b',dmatch.group(1))) if dmatch else 0
    if nd!=4:errors.append(f'{p.name}: expected 4 deep exercises, found {nd}')
    if re.search(r'TODO\[|TODO\b|图占位|待实跑|实跑后填',plain):errors.append(f'{p.name}: unresolved placeholder')
    chapters.append({'file':str(p.relative_to(ROOT)),'numbered_sections':nsec,'CJK_characters_excluding_code':len(re.findall(r'[\u4e00-\u9fff]',plain)),'deep_exercises':nd})
if len(chapters)!=17:errors.append('Expected 17 chapters')
if sum(c['numbered_sections'] for c in chapters)!=119:errors.append('Expected 119 numbered sections')
nb_report=json.loads((ROOT/'review_v3/notebook_execution.json').read_text())
notebooks=[]
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(p,as_version=4);code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    digest=hashlib.sha256(code.encode()).hexdigest();rec=nb_report.get(p.name,{})
    errs=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
    if errs or rec.get('status')!='passed' or digest!=rec.get('code_sha256'):errors.append(f'{p.name}: execution/source mismatch')
    notebooks.append({'file':str(p.relative_to(ROOT)),'code_sha256':digest,'recorded_status':rec.get('status'),'error_outputs':len(errs)})
legacy=json.loads((ROOT/'review_v3/chapter03_legacy_notebook_execution.json').read_text())
for name,rec in legacy.items():
    p=ROOT/name
    if not p.exists() or rec.get('status')!='passed':errors.append('Legacy notebook failed: '+name)
    else:
        nb=nbformat.read(p,as_version=4)
        if any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[])):errors.append('Legacy output error: '+name)
num=json.loads((ROOT/'review_v3/numerical_audit.json').read_text())
if num['passed']!=num['total']:errors.append('Numerical audit has failures')
experiment=json.loads((ROOT/'review_v3/experiment_execution.json').read_text())
if len(experiment)!=18 or any(v['exit_code']!=0 for v in experiment.values()):errors.append('Experiment process record incomplete')
figures=sorted(set(p.stem for p in (ROOT/'figures').iterdir() if p.suffix in ['.png','.pdf']))
out={'status':'passed' if not errors else 'failed','errors':errors,'chapters':chapters,'chapter_count':len(chapters),'numbered_sections':sum(c['numbered_sections'] for c in chapters),'chapter_CJK_chars_excluding_code':sum(c['CJK_characters_excluding_code'] for c in chapters),'deep_exercises':sum(c['deep_exercises'] for c in chapters),'resolved_TeX_inputs':[str(p.relative_to(ROOT)) for p in visited],'appendix_chapters':len(re.findall(r'\\chapter\{',text.split('\\appendix')[-1])),'unique_figure_stems':len(figures),'new_v3_figure_stems':sum(s.startswith('v3_') for s in figures),'referenced_graphic_files':len(set(assets)),'notebooks':notebooks,'executed_notebook_count':len(notebooks)+len(legacy),'numerical_checks':{'passed':num['passed'],'total':num['total']},'fresh_process_experiments':len(experiment)}
try:
    import fitz
    out['main_pdf_pages']=len(fitz.open(ROOT/'main.pdf'));out['teacher_pdf_pages']=len(fitz.open(ROOT/'teacher_manual.pdf'))
except ImportError:out['pdf_page_counts']='PyMuPDF not installed; optional inspection omitted.'
(ROOT/'review_v3/structure_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ['chapters','notebooks','resolved_TeX_inputs']},ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
