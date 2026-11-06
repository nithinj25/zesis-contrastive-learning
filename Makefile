# TRACE build targets (CLAUDE.md section 19.13). On Windows without make, run the commands by hand.
PY ?= python

.PHONY: data test audit probe

data:            ## raw/ -> data/meta.parquet + processed/ (about 1 hour on 6 CPU workers)
	$(PY) -m data.ingest
	$(PY) -m data.dedupe
	$(PY) -m data.process
	$(PY) -m data.quality
	$(PY) -m data.splits

audit:           ## leaks, drops and a contact sheet
	$(PY) scripts/dedupe_audit.py

probe:           ## camera signal before / after preprocessing
	$(PY) scripts/camera_probe.py

test:
	$(PY) -m pytest -q
