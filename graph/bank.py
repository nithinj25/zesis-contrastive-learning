"""Disease bank: one prototype vector per disease ("Situation A" in CLAUDE.md section 4; section 9.8 image(d)).

A prototype is the L2-normalised mean of a disease's frozen global features. A new photo is compared to every
prototype by cosine; adding a disease means adding one vector, with no retraining:

    bank = Bank("trace_v2")
    bank.add("RP", feats_of_rp_photos, ids_of_rp_photos)
    bank.rank(x)            # [(disease, cosine), ...] best first
    bank.save("artifacts/trace_v2/bank.npz")

Built by `python -m graph.bank --name trace_v2`: common diseases from head-training photos, rare diseases (the
leave-disease-out targets) from their RFMiD + RFMiD 2.0 photos. JSIEC never enters the bank: it is the test set.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

MIN_PHOTOS = 5


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Bank:
    def __init__(self, encoder: str):
        self.encoder, self.names, self.protos, self.n, self.members = encoder, [], [], [], {}

    def add(self, name: str, feats: np.ndarray, ids=None):
        """Add (or replace) one disease from its photos' features (N, D)."""
        if name in self.names:
            self.remove(name)
        self.names.append(name)
        self.protos.append(unit(feats.mean(0)))
        self.n.append(len(feats))
        self.members[name] = list(ids) if ids is not None else []

    def remove(self, name: str):
        i = self.names.index(name)
        for lst in (self.names, self.protos, self.n):
            lst.pop(i)
        self.members.pop(name, None)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Cosine of each row of X (N, D) to every prototype -> (N, n_diseases)."""
        return unit(np.atleast_2d(X)) @ np.stack(self.protos).T

    def rank(self, x: np.ndarray, k: int | None = None) -> list[tuple[str, float]]:
        s = self.scores(x)[0]
        order = np.argsort(-s)[:k]
        return [(self.names[i], float(s[i])) for i in order]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, encoder=self.encoder, names=np.array(self.names), protos=np.stack(self.protos), n=np.array(self.n),
                 members=np.array([f"{d}:{i}" for d, ids in self.members.items() for i in ids]))

    @classmethod
    def load(cls, path: str) -> "Bank":
        z = np.load(path, allow_pickle=False)
        b = cls(str(z["encoder"]))
        b.names, b.protos, b.n = list(z["names"]), list(z["protos"]), [int(v) for v in z["n"]]
        for m in z["members"]:
            d, i = str(m).split(":", 1)
            b.members.setdefault(d, []).append(i)
        return b


def disease_rows(meta: pd.DataFrame, rare: list[str]) -> dict:
    """image ids per disease: common diseases from head training, rare ones from RFMiD + RFMiD 2.0 (never JSIEC)."""
    head = meta[meta.in_heads & meta.label_ok]
    counts = pd.Series([d for x in head.diseases for d in x]).value_counts()
    out = {d: list(head.index[head.diseases.map(lambda x, d=d: d in list(x))]) for d, n in counts.items()
           if n >= MIN_PHOTOS and d not in rare}
    for d in rare:
        sel = meta[(meta.ldo_disease == d) & meta.label_ok & (meta.bucket != "excluded") & meta.dataset.isin(["rfmid", "rfmid2"])]
        out[d] = list(sel.index)
    return out


if __name__ == "__main__":
    import yaml

    from eval.probes import load_features

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    rare = yaml.safe_load(open("data/ontology.yaml", encoding="utf-8"))["leave_disease_out"]
    f = load_features(args.name)
    row = f.set_index("image_id").row
    X = unit(f.attrs["X"])
    bank = Bank(args.name)
    for d, ids in disease_rows(meta, rare).items():
        ids = [i for i in ids if i in row.index]
        bank.add(d, X[row.loc[ids].values], ids)
    path = os.path.join("artifacts", args.name, "bank.npz")
    bank.save(path)
    print(f"saved {path}: " + ", ".join(f"{d} ({n})" for d, n in zip(bank.names, bank.n)))
