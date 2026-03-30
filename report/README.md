### Report generation using LaTeX

To compile the report, ensure a LaTeX distribution is installed:

- macOS: MacTeX (recommended) or BasicTeX
- Windows: MiKTeX
- Linux: TeX Live

Then run:

`pdflatex report.tex`

---

For references (if using BibTeX):

\* and yes you do need to run `pdflatex` 3 times in total.

- `pdflatex report.tex`

- `bibtex report`

- `pdflatex report.tex`

- `pdflatex report.tex`

---
Alternatively, you can use latexmk (recommended):

latexmk -pdf report.tex