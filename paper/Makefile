# GeomTok paper builds. Requires: tectonic (brew install tectonic).
MAIN = acl_latex

.PHONY: review preprint final clean
review:                 ## anonymous, line-numbered — submit THIS to ARR/OpenReview
	tectonic $(MAIN).tex
	@echo ">> $(MAIN).pdf  (anonymous review build)"

preprint:               ## non-anonymous + page numbers — for arXiv
	sed 's/\[review\]{acl}/[preprint]{acl}/' $(MAIN).tex > _build_preprint.tex
	tectonic _build_preprint.tex
	mv _build_preprint.pdf $(MAIN)_preprint.pdf
	rm -f _build_preprint.tex
	@echo ">> $(MAIN)_preprint.pdf  (non-anonymous)"

final:                  ## camera-ready (after acceptance)
	sed 's/\[review\]{acl}/[final]{acl}/' $(MAIN).tex > _build_final.tex
	tectonic _build_final.tex
	mv _build_final.pdf $(MAIN)_final.pdf
	rm -f _build_final.tex
	@echo ">> $(MAIN)_final.pdf  (camera-ready)"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out _build_*.tex
