.PHONY: all pdf teacher experiments figs check notebooks execute clean
all: pdf teacher
pdf:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
teacher:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error teacher_manual.tex
experiments:
	python scripts/run_deep_experiments.py
figs: experiments
check:
	python scripts/audit_numerics_v3.py
	python scripts/audit_structure_v3.py
notebooks:
	python scripts/build_notebooks_v3.py
execute:
	python scripts/execute_notebooks_v3.py
clean:
	latexmk -c main.tex
	latexmk -c teacher_manual.tex
