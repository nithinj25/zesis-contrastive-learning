"""One photo in, one decision out (CLAUDE.md 3.1, 10.5): the full TRACE pipeline for a single fundus photo.

    pipe = TracePipeline()                      # loads artifacts/trace_v2/ and checkpoints/trace_v2/encoder.pt
    r = pipe.analyse(rgb_uint8, camera=None)    # dict: decision, findings, evidence, similar cases, flags

Reuses the evaluation code unchanged (encoder.export, findings.crops, findings.export, trust.*, graph.score,
eval.rare_bank.rare_margin), so the demo makes the same decision the reported results were computed with:
  quality gate -> OOD (10th-neighbour cosine distance, per-camera threshold) -> rare-disease bank (margin tau) ->
  per-camera conformal set of the calibrated 7-class classifier -> trust.decide.decide (disagreement flag off,
  as decided on val-A). Findings and the knowledge graph give the evidence for / against; they do not decide.
Needs only artifacts/<name>/ (built by make heads / make trust / python -m demo.bundle) and the encoder weights;
similar-case thumbnails are shown only when processed/ exists locally (photos are not redistributed).
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import torch

from data.preprocess import correct_illumination, find_fov, standardise_geometry
from encoder.export import finish_view, load_trace_encoder
from eval.rare_bank import rare_margin
from findings.crops import crops_for_photo
from findings.export import finding_scores, load_models
from graph.bank import Bank, unit
from graph.score import explain, score
from trust.classifier import CLASSES
from trust.decide import decide
from trust.ood import KnnOOD

NAMES = {"normal": "Normal", "DR": "Diabetic retinopathy", "myopia": "Pathological myopia", "ARMD": "Age-related macular degeneration",
         "BRVO": "Branch retinal vein occlusion", "CSR": "Central serous retinopathy", "other_abnormal": "Abnormal, no listed disease",
         "AION": "Anterior ischaemic optic neuropathy", "CRVO": "Central retinal vein occlusion", "RP": "Retinitis pigmentosa",
         "hypertensive_retinopathy": "Hypertensive retinopathy"}


class TracePipeline:
    def __init__(self, name: str = "trace_v2", weights: str | None = None, device: str = "cpu"):
        art = os.path.join("artifacts", name)
        self.device = device
        self.stats = json.load(open("data/trace_preprocess.json"))
        self.enc = load_trace_encoder(weights or os.path.join("checkpoints", name, "encoder.pt"), device)
        self.gate = joblib.load(os.path.join(art, "quality.joblib"))
        trust = joblib.load(os.path.join(art, "trust.joblib"))
        self.clf, self.mond = trust["classifier"], trust["mondrian"]
        self.fj = joblib.load(os.path.join(art, "findings.joblib"))
        self.fmodels = load_models(name, device)
        b = joblib.load(os.path.join(art, "demo_bundle.joblib"))
        self.ood = KnnOOD().fit(b["ood"]["X"].astype(np.float32), b["ood"]["ids"])
        th = b["ood"]["thresholds"]
        self.ood.thr_all, self.ood.thr = th["pooled"], {k: v for k, v in th.items() if k != "pooled"}
        self.bank = Bank(name)
        self.bank.names, self.bank.protos, self.bank.n = list(b["bank"]["names"]), list(b["bank"]["protos"]), list(b["bank"]["n"])
        self.rare, self.tau = b["bank"]["rare"], b["bank"]["tau"]
        self.lib = b["library"]
        self.lib_X = unit(self.lib["X"].astype(np.float32))
        self.G = b["graph"]
        self.cameras = b["cameras"]

    # ---- steps ----------------------------------------------------------------------------------------------
    def process(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw photo -> processed 512 px photo + retina mask (steps 3, 4, 6). ValueError if no retina is found."""
        fov = find_fov(rgb)
        if fov is None:
            raise ValueError("retina not found")
        std, mask = standardise_geometry(rgb, fov)
        return correct_illumination(std, mask, strength=self.stats["params"]["illumination_strength"]), mask

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> np.ndarray:
        return self.enc(x[None].to(self.device)).cpu().numpy().astype(np.float32)

    def analyse(self, rgb: np.ndarray | None = None, camera: str | None = None, processed=None) -> dict:
        """rgb: raw RGB uint8 photo; or processed=(rgb512, mask) to skip steps 3-6. camera: a known camera id or None."""
        camera = camera if camera in self.cameras else None
        try:
            lit, mask = processed if processed is not None else self.process(rgb)
        except ValueError as e:
            return {"action": "retake", "reason": f"{e}: please retake the photo", "diseases": [], "retina_found": False}
        g = self._encode(finish_view(lit, mask, camera, self.stats, 384))
        disc, _, lm = crops_for_photo(lit, mask, camera, self.stats)
        d = self._encode(disc) if lm.ok else None

        p_rej = float(self.gate.p_reject(g)[0])
        retake = bool(self.gate.retake(g)[0])
        q = {f: float(v[0]) for f, v in finding_scores(self.fmodels, self.fj, g, d).items()}

        group = np.array([camera or "unknown"])
        p = self.clf.proba(g)[0]
        cset = [CLASSES[j] for j in np.where(self.mond.sets(p[None], group)[0])[0]]
        ood_score = float(self.ood.score(g)[0])
        ood_thr = float(self.ood.threshold(group)[0])
        s = self.bank.scores(g)[0]
        is_rare = np.isin(self.bank.names, self.rare)
        margin = float(rare_margin(self.bank, g, self.rare)[0])
        top_rare = np.array(self.bank.names)[is_rare][s[is_rare].argmax()]
        rare_call = top_rare if margin > self.tau else None
        dec = decide(retake, ood_score > ood_thr, rare_call, cset, disagree=False)

        # evidence for / against (knowledge graph on the calibrated findings; explanation only)
        G = self.G
        qv = np.array([q.get(f, np.nan) for f in G["findings"]])[None]
        _, post, terms = score(qv, G["p"], G["p_not"], G["prior"], G["r"])
        shown = [x for x in (dec.diseases or [CLASSES[int(p.argmax())]]) if x in G["diseases"]][:3]
        evidence = {x: dict(zip(("for", "against"), explain(terms[0, G["diseases"].index(x)], G["findings"]))) for x in shown}

        sims = self.lib_X @ unit(g)[0]
        top = np.argsort(-sims)[:5]
        cases = [{"id": str(self.lib["ids"][i]), "diagnosis": str(self.lib["dx"][i]), "cosine": float(sims[i]),
                  "processed_path": str(self.lib["processed_path"][i])} for i in top]
        return {
            "action": dec.action, "reason": dec.reason, "diseases": dec.diseases, "retina_found": True,
            "processed": (lit, mask), "landmarks": lm, "camera": camera,
            "quality": {"p_reject": p_rej, "retake": retake},
            "ood": {"score": ood_score, "threshold": ood_thr, "unfamiliar": ood_score > ood_thr},
            "rare": {"margin": margin, "tau": self.tau, "top": str(top_rare), "called": rare_call},
            "classifier": dict(zip(CLASSES, map(float, p))), "conformal_set": cset,
            "findings": dict(sorted(q.items(), key=lambda t: -np.nan_to_num(t[1], nan=-1))),
            "reliability": {f: r["reliability"] for f, r in self.fj["report"].items()},
            "evidence": evidence, "graph_posterior": dict(zip(G["diseases"], map(float, post[0]))),
            "similar_cases": cases, "feature": g[0],
        }
