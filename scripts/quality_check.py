"""Static text, figure, plan and artifact checks; not a scientific peer review."""
from __future__ import annotations
import re, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PL=json.loads((ROOT/'docs/writing_plan_extracted.json').read_text())
CJK=re.compile(r'[\u3400-\u9fff]')
sections=0
chars=0
issues=[]
records=[]
all_refs=[]
for ch in PL:
    n=int(ch['chapter'])
    f=ROOT/f'chapters/ch{n:02}.tex'
    if not f.exists():issues.append(f'Missing {f}');continue
    s=f.read_text(encoding='utf-8')
    parts=re.findall(r'\\section\{([^}]+)\}',s)
    if len(parts)!=len(ch['sections']):
        issues.append(f'ch{n:02}: {len(parts)} sections but plan has {len(ch["sections"])}')
    if len(re.findall(r'\\chapter\{',s))!=1:
        issues.append(f'ch{n:02}: not exactly one chapter')
    if n!=3 and '\\chapterexercise' not in s:issues.append(f'ch{n:02}: no practice questions')
    if n!=3 and '\\chapterpractice' not in s:issues.append(f'ch{n:02}: no practice description')
    figs=re.findall(r'\\fig(?:\[[^]]+\])?\{([^}]+)\}',s)
    fullfigs=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s)
    for p in figs:
        x=ROOT/'figures'/p
        if not x.exists():issues.append(f'missing figure {x}')
    for p in fullfigs:
        x=ROOT/p
        if not x.exists():issues.append(f'missing graphics {x}')
    c=len(CJK.findall(s));chars+=c;sections+=len(parts)
    records.append(dict(chapter=n,sections=len(parts),characters=c,figures=len(figs)+len(fullfigs)))
    if not (ROOT/f'notebooks/ch{n:02}_student.ipynb').exists():issues.append(f'missing student notebook {n}')
    if not (ROOT/f'notebooks/ch{n:02}_teacher.ipynb').exists():issues.append(f'missing teacher notebook {n}')
if sections!=119:issues.append(f'Total section count {sections}, expected 119')
if sum(ch['hours'] for ch in PL)!=64:issues.append('Hours not 64')
if not (ROOT/'main.pdf').exists(): issues.append('Main book PDF missing')
if not (ROOT/'teacher_manual.pdf').exists(): issues.append('Teacher PDF missing')
num_pdf_figs=len(list((ROOT/'figures').glob('*.pdf')))
num_png_figs=len(list((ROOT/'figures').glob('*.png')))
if num_pdf_figs<30 or num_png_figs<35: issues.append('Too few figure artifacts')
if '第17章' not in (ROOT/'docs/teacher_manual.tex').read_text():issues.append('Teacher key incomplete')
report={'status':'PASS' if not issues else 'FAIL','chapters':len(records),'sections':sections,
        'hours':sum(ch['hours'] for ch in PL),'chapter_hanzi_count':chars,
        'planned_hanzi_approx':sum(ch['thousand_chars'] for ch in PL)*1000,
        'pdf_vector_figures':num_pdf_figs,'png_figures':num_png_figs,
        'student_notebooks':len(list((ROOT/'notebooks').glob('ch*_student.ipynb'))),
        'teacher_notebooks':len(list((ROOT/'notebooks').glob('ch*_teacher.ipynb'))),
        'chapters_detail':records,'issues':issues}
(ROOT/'review/quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='chapters_detail'},ensure_ascii=False,indent=2))
if issues:sys.exit(1)
