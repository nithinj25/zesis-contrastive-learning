"""TRACE demo (CLAUDE.md 19.10): upload a fundus photo, get the decision card.

    python -m demo.app            # then open http://127.0.0.1:7860   (CPU is enough: ~1 s per photo)

Panels: decision card (diagnose / short-list / refer / retake, with the reason and flags), the processed photo with
the disc and macula boxes, calibrated findings, disease scores with the evidence for and against, and the 5 most
similar confirmed cases (thumbnails only if processed/ exists locally). Runs locally only: no share link, Gradio
usage analytics off. Research prototype, not a medical device.
"""
from __future__ import annotations

import html
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import gradio_client.utils as _gcu
import numpy as np

# Gradio 5.20 + pydantic >= 2.11: JSON schemas may contain bare `true` entries, which gradio_client's type printer
# cannot read, so every page request fails. Treat them as "Any" (only affects API type labels, not the demo).
_get_type, _to_py = _gcu.get_type, _gcu._json_schema_to_python_type
_gcu.get_type = lambda schema: "Any" if isinstance(schema, bool) else _get_type(schema)
_gcu._json_schema_to_python_type = lambda schema, defs=None: "Any" if isinstance(schema, bool) else _to_py(schema, defs)

from data.process import load_processed
from demo.pipeline import NAMES, TracePipeline
from findings.localise import DD

PIPE = TracePipeline(device="cuda" if os.environ.get("TRACE_DEMO_GPU") else "cpu")
UNKNOWN = "Unknown camera"
CAMERA_LABELS = {UNKNOWN: None, **{c: c for c in PIPE.cameras}}
ACTION = {"diagnose": ("DIAGNOSE", "#1E5AA8"), "shortlist": ("SHORT-LIST for clinician review", "#B85C1A"),
          "refer": ("REFER to a specialist", "#9B2C2C"), "retake": ("RETAKE the photo", "#52514e")}
FINDING_NAMES = lambda f: f.replace("_", " ")


def overlay(lit: np.ndarray, mask: np.ndarray, lm) -> np.ndarray:
    img = (lit * mask[..., None]).astype(np.uint8).copy()
    if lm.ok:
        for (x, y), side, col, name in (((lm.disc_x, lm.disc_y), 2 * DD, (255, 255, 255), "disc"),
                                         ((lm.fovea_x, lm.fovea_y), 3 * DD, (255, 220, 120), "macula")):
            p0 = (int(x - side / 2), int(y - side / 2))
            cv2.rectangle(img, p0, (int(x + side / 2), int(y + side / 2)), col, 2)
            cv2.putText(img, name, (p0[0] + 4, p0[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


def card(r: dict) -> str:
    title, colour = ACTION[r["action"]]
    names = ", ".join(NAMES.get(d, d) for d in r["diseases"]) or "—"
    rows = [f"<div style='display:inline-block;background:{colour};color:#fff;font-size:18px;font-weight:600;"
            f"padding:6px 12px;border-radius:8px'>{title}</div>",
            f"<div style='font-size:17px;margin:6px 0'>{html.escape(names) if r['action'] != 'retake' else ''}</div>",
            f"<div style='color:var(--body-text-color-subdued)'>Why: {html.escape(r['reason'])}</div>"]
    if r.get("retina_found"):
        flags = []
        if r["quality"]["retake"]:
            flags.append(f"⚠ Photo quality low (p reject {r['quality']['p_reject']:.2f})")
        if r["ood"]["unfamiliar"]:
            flags.append(f"⚠ Unfamiliar photo (distance {r['ood']['score']:.3f} > {r['ood']['threshold']:.3f})")
        if r["rare"]["called"]:
            flags.append(f"⚠ Rare-disease bank: closest to {NAMES.get(r['rare']['called'])} (margin {r['rare']['margin']:+.3f} > {r['rare']['tau']:.3f})")
        flags.append("90% conformal set: " + (", ".join(NAMES.get(c, c) for c in r["conformal_set"]) or "empty"))
        flags.append(f"Camera: {r['camera'] or 'unknown (no colour correction)'}")
        rows.append("<ul style='margin:8px 0 0 18px;color:var(--body-text-color)'>" + "".join(f"<li>{html.escape(f)}</li>" for f in flags) + "</ul>")
    rows.append("<div style='margin-top:10px;font-size:12px;color:var(--body-text-color-subdued)'>Research prototype · not a medical device · "
                "every output must be read by an eye-care professional</div>")
    return ("<div style='border:1px solid var(--border-color-primary);border-radius:12px;padding:16px;"
            "color:var(--body-text-color)'>" + "".join(rows) + "</div>")


MIN_EVIDENCE = 0.1  # display only: weaker terms (likelihood ratio within ~10% of 1) say nothing worth showing


def evidence_md(r: dict) -> str:
    out = []
    for d, e in r["evidence"].items():
        pro = [(f, v) for f, v in e["for"] if v >= MIN_EVIDENCE]
        con = [(f, v) for f, v in e["against"] if v <= -MIN_EVIDENCE]
        line = f"**{NAMES.get(d, d)}**  \n"
        line += "✔ " + (", ".join(f"{FINDING_NAMES(f)} ({v:+.2f})" for f, v in pro) or "no strong supporting finding") + "  \n"
        line += "✘ " + (", ".join(f"{FINDING_NAMES(f)} ({v:+.2f})" for f, v in con) or "no strong finding against")
        out.append(line)
    note = ("\n\n*Evidence = weighted log-likelihood ratio of each detected finding under the knowledge graph (positive = "
            f"for; only |value| ≥ {MIN_EVIDENCE} shown). It explains the decision; it does not make it. "
            "AION / CRVO edges are literature drafts, not yet clinician-reviewed.*")
    return ("\n\n".join(out) or "No disease to explain.") + note


def similar(r: dict):
    gal = []
    for c in r["similar_cases"]:
        path = os.path.join("processed", c["processed_path"])
        caption = f"{NAMES.get(c['diagnosis'], c['diagnosis'])} · cosine {c['cosine']:.3f}"
        if os.path.exists(path):
            rgb, mask = load_processed(path)
            gal.append((cv2.resize(rgb * mask[..., None], (192, 192), interpolation=cv2.INTER_AREA), caption))
        else:
            gal.append((np.full((192, 192, 3), 235, np.uint8), caption + " (photo not available)"))
    return gal


def run(photo, camera_label):
    if photo is None:
        return "<i>Upload a fundus photo.</i>", None, {}, {}, "", []
    r = PIPE.analyse(photo, CAMERA_LABELS.get(camera_label))
    if not r.get("retina_found"):
        return card(r), None, {}, {}, "", []
    lit, mask = r["processed"]
    findings = {FINDING_NAMES(f): v for f, v in r["findings"].items() if not np.isnan(v)}
    scores = {NAMES.get(k, k): v for k, v in r["classifier"].items()}
    return card(r), overlay(lit, mask, r["landmarks"]), findings, scores, evidence_md(r), similar(r)


def examples() -> list:
    """A few local RFMiD test photos as one-click examples, if the raw data is present."""
    d = os.path.join("raw", "rfmid", "Test_Set", "Test_Set", "Test")
    return [[os.path.join(d, f"{i}.png"), UNKNOWN] for i in (1, 5, 12, 40) if os.path.exists(os.path.join(d, f"{i}.png"))]


with gr.Blocks(title="TRACE demo") as app:
    gr.Markdown("## TRACE: Trustworthy Retinal Contrastive Encoder\nUpload a colour fundus photo. TRACE checks the photo, "
                "describes what it sees, and decides whether to **diagnose**, **short-list** or **refer**.")
    with gr.Row():
        with gr.Column(scale=1):
            photo = gr.Image(type="numpy", label="Fundus photo", height=320)
            camera = gr.Dropdown(list(CAMERA_LABELS), value=UNKNOWN, label="Camera (if known)")
            go = gr.Button("Analyse", variant="primary")
            ex = examples()
            if ex:
                gr.Examples(ex, [photo, camera], label="Example photos (RFMiD test set, local only)")
        with gr.Column(scale=1):
            decision = gr.HTML()
            view = gr.Image(label="Processed photo: disc and macula", height=320, interactive=False)
    with gr.Row():
        findings = gr.Label(num_top_classes=8, label="Findings (calibrated probability)")
        diseases = gr.Label(num_top_classes=7, label="Disease scores (calibrated classifier)")
    evidence = gr.Markdown(label="Evidence for / against")
    cases = gr.Gallery(label="5 most similar confirmed cases", columns=5, height=240)
    go.click(run, [photo, camera], [decision, view, findings, diseases, evidence, cases])

if __name__ == "__main__":
    # show_api=False: Gradio 5.20 + pydantic 2.11 crash while building the API page (gradio_client get_type), which
    # makes Gradio think localhost is unreachable. The demo does not need the API page.
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, show_api=False)
