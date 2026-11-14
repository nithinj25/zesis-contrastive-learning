# TRACE build targets (CLAUDE.md section 19.13). `make help` lists them.
# Windows without make: run the indented commands by hand from the repo root (same order).
# ENC = the encoder every head / trust / eval step reads (features/<ENC>/); W = its backbone weights.
PY   ?= python
ENC  ?= trace_v2
W    ?= checkpoints/$(ENC)/encoder.pt
BASE ?= imagenet_r50 moco_plain_e15 supervised_r50
IMAGES ?= photos
OUT  ?= features.npy

.PHONY: help data audit probe pretrain baselines features heads trust eval figures demo embed test

help:            ## list targets
	@grep -E '^[a-z]+:.*## ' Makefile | sed -E 's/:.*## /\t/'

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

pretrain:        ## the TRACE encoder: 224 px phase, then 384 px phase, val-A-best copied to checkpoints/trace_v2 (~12 GPU-h)
	$(PY) -m encoder.train --config-name pretrain
	$(PY) -m encoder.train --config-name pretrain_384
	$(PY) -c "import os, shutil; os.makedirs('checkpoints/trace_v2', exist_ok=True); shutil.copyfile('checkpoints/trace_v1_384/encoder_best.pt', 'checkpoints/trace_v2/encoder.pt')"

baselines:       ## plain MoCo v2 (15 epochs) and the supervised ResNet-50, then their features (~2 GPU-h)
	$(PY) -m encoder.train --config-name pretrain_plain encoder_name=moco_plain_e15 optim.epochs=15 optim.warmup_epochs=2
	$(PY) -m encoder.supervised
	$(PY) -m encoder.features --name imagenet_r50
	$(PY) -m encoder.features --name moco_plain_e15 --weights checkpoints/moco_plain_e15/encoder.pt
	$(PY) -m encoder.features --name supervised_r50 --weights checkpoints/supervised_r50/encoder.pt

features:        ## frozen features of every photo for ENC (+ disc / macula crops), e.g. make features ENC=trace_v2
	$(PY) -m encoder.features --name $(ENC) --weights $(W)
	$(PY) -m findings.crops --name $(ENC) --weights $(W)

heads:           ## finding detectors and the rare-disease bank on ENC's frozen features (minutes, CPU)
	$(PY) -m findings.probes --name $(ENC)
	$(PY) -m graph.bank --name $(ENC)

trust:           ## classifier + temperature, per-camera conformal, OOD index, quality gate -> artifacts/ENC/
	$(PY) -m eval.trust_eval --name $(ENC)
	$(PY) -m eval.quality_eval --name $(ENC)

eval:            ## every result in MODEL_CARD.md for ENC against BASE -> outputs/eval/ (CPU, about 1-2 h)
	$(PY) -m eval.probes --name $(ENC)
	$(PY) -m eval.leave_camera_out $(ENC) $(BASE)
	$(PY) -m eval.fewshot $(ENC) $(BASE)
	$(PY) -m eval.compare $(ENC) $(BASE)
	$(PY) -m eval.retrieval --encoders $(BASE) $(ENC) --figure $(ENC)
	$(PY) -m eval.rare_bank $(ENC) $(BASE)
	$(PY) -m eval.dist_calib_eval --name $(ENC) --final
	$(PY) -m eval.leave_disease_out --name $(ENC)
	$(PY) -m eval.site_onboarding --name $(ENC)
	$(PY) -m eval.decision_eval --name $(ENC)
	$(PY) -m encoder.lora --encoder $(ENC)

figures:         ## UMAP (disease / camera), the disease x finding map and risk-coverage curves -> outputs/figures/
	$(PY) -m scripts.umap_plots --encoders imagenet_r50 $(ENC)
	$(PY) scripts/disease_finding_map.py --name $(ENC)
	$(PY) -m scripts.risk_coverage_plot

demo:            ## build the demo files, then the Gradio demo on http://127.0.0.1:7860 (CPU is enough)
	$(PY) -m findings.export --name $(ENC)
	$(PY) -m demo.bundle --name $(ENC)
	$(PY) -m demo.app

embed:           ## features of any folder of photos: make embed IMAGES=path/to/photos OUT=features.npy (add CAMERA=<id> if known)
	$(PY) -m encoder.embed --images $(IMAGES) --out $(OUT) --weights $(W) $(if $(CAMERA),--camera $(CAMERA))

test:            ## unit tests
	$(PY) -m pytest -q
