Build (Overleaf)
1) Create a new Overleaf project.
2) Upload: main.tex, refs.bib, and the figures/ folder.
3) Set main file to main.tex and compile.

Build (Local TeX Live / MiKTeX)
1) Open a terminal in this folder (paper/)
2) Run:
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex

Notes
- This paper uses the IEEEtran conference format.
- Figures are included from paper/figures.
