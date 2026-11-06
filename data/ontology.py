"""Map each photo's dataset labels to TRACE findings and diseases, using data/ontology.yaml (section 5.6)."""
from __future__ import annotations

import yaml

ONTOLOGY_PATH = "data/ontology.yaml"


def load(path: str = ONTOLOGY_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _index(section: dict) -> dict[tuple[str, str], list[str]]:
    """(dataset, label) -> names that label counts towards."""
    out: dict[tuple[str, str], list[str]] = {}
    for name, entry in section.items():
        for dataset, labels in entry.get("labels", {}).items():
            for label in labels:
                out.setdefault((dataset, label), []).append(name)
    return out


class Ontology:
    def __init__(self, data: dict):
        self.data = data
        self.findings = _index(data["findings"])
        self.diseases = _index(data["diseases"])
        self.leave_out = list(data.get("leave_disease_out", []))

    def map(self, dataset: str, labels, label_ok: bool, disease_risk=None) -> tuple[list[str], list[str]]:
        """Returns (findings, diseases) for one photo. Unknown labels map to nothing, never to a guess."""
        if not label_ok:
            return [], []
        labels = list(labels)
        findings = sorted({f for lab in labels for f in self.findings.get((dataset, lab), [])})
        diseases = sorted({d for lab in labels for d in self.diseases.get((dataset, lab), [])})
        # RFMiD marks a normal eye by having no labels at all and Disease_Risk 0.
        if dataset == "rfmid" and not labels and disease_risk == 0:
            diseases = ["normal"]
        return findings, diseases
