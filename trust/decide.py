"""Decision rule (CLAUDE.md sections 10.5, 10.6): one photo in, one of four actions out.

Checks run in order; an earlier one stops the later ones:
  1 quality gate says unreadable            -> RETAKE
  2 OOD: unfamiliar photo                   -> REFER  ("unfamiliar", bank short-list attached)
  3 rare-disease bank fires (margin rule)   -> REFER  ("possible <rare disease>"; a rare suspicion is never diagnosed
                                                       by a screening tool, it is sent to a specialist)
  4 conformal set is exactly {other_abnormal} -> REFER ("abnormal signs, no matching disease"; findings listed)
    conformal set has exactly 1 disease     -> DIAGNOSE, unless the findings disagree (10.6) -> SHORTLIST
  5 set has 2-3 diseases                    -> SHORTLIST
  6 set empty or > 3 diseases               -> REFER
"""
from __future__ import annotations

from dataclasses import dataclass, field

RETAKE, REFER, SHORTLIST, DIAGNOSE = "retake", "refer", "shortlist", "diagnose"


@dataclass
class Decision:
    action: str
    reason: str
    diseases: list = field(default_factory=list)


def decide(retake: bool, unfamiliar: bool, rare: str | None, conformal_set: list, disagree: bool = False) -> Decision:
    if retake:
        return Decision(RETAKE, "photo quality too low to read")
    if unfamiliar:
        return Decision(REFER, "unfamiliar photo: nothing like it among known cases")
    if rare:
        return Decision(REFER, f"possible {rare}: rare disease, specialist review", [rare])
    n = len(conformal_set)
    if list(conformal_set) == ["other_abnormal"]:
        return Decision(REFER, "abnormal signs, no matching disease: specialist review", list(conformal_set))
    if n == 1 and not disagree:
        return Decision(DIAGNOSE, "one disease fits at 90% confidence", list(conformal_set))
    if n == 1 and disagree:
        return Decision(SHORTLIST, "evidence disagrees: detected findings do not support the image match", list(conformal_set))
    if 2 <= n <= 3:
        return Decision(SHORTLIST, f"{n} diseases fit: clinician review", list(conformal_set))
    return Decision(REFER, "no clear answer" if n == 0 else f"{n} diseases fit: too uncertain", list(conformal_set))
