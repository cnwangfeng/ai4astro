#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
  latexmk -xelatex -interaction=nonstopmode -halt-on-error teacher_manual.tex
else
  for name in main teacher_manual; do
    xelatex -interaction=nonstopmode -halt-on-error "$name.tex"
    xelatex -interaction=nonstopmode -halt-on-error "$name.tex"
    xelatex -interaction=nonstopmode -halt-on-error "$name.tex"
  done
fi
