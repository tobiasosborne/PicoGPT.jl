"""Recompute local gate certificates and recount the saved stock screens."""
import itertools
import json
from pathlib import Path

artifacts = Path(__file__).resolve().parents[2] / "artifacts"


def runs(name):
    return json.loads((artifacts / (name + ".json")).read_text())["runs"]


def logit(params, current, left):
    width = (len(params) - 1) // 4
    hidden = [
        max(0, params[i] * current + params[width + i] * left + params[2 * width + i])
        for i in range(width)
    ]
    return params[-1] + sum(w * h for w, h in zip(params[3 * width:-1], hidden))


def gate_margins(name):
    margins = []
    for run in runs(name):
        params = run["θ"]
        values = [-x * y * logit(params, x, y) for x, y in itertools.product((-1, 1), repeat=2)]
        if gate := run.get("gate"):
            assert max(abs(a - b) for a, b in zip(sorted(values), sorted(gate["margins"]))) < 1e-12
        margins.append(min(values))
    return margins


for task in ("prefix", "rule60"):
    for model in ("compact-tied", "three-tied", "tied"):
        name = f"round2-{task}-{model}"
        margins = gate_margins(name)
        print(name, "minimum margin by seed", margins)
        if model != "three-tied":
            assert min(margins) > 1

# Three possible symbols for the 13-parameter Rule 60 cell in seed 2.
# Zero register values are named by their parents' bits.
params = runs("round2-rule60-three-tied")[1]["θ"]
z00, z11, one = params[-1], -1, 1
for curr, left, expected in (
    (z11, z11, z00), (z00, z00, z00), (z00, z11, z00),
    (one, z11, one), (one, z00, one), (z11, one, one), (z00, one, one),
    (one, one, z11)
):
    assert min(1, max(-1, logit(params, curr, left))) == expected
print("Unreachable interior pair (current z11, left z00) logit", logit(params, z11, z00))

def within_floor(run):
    return run["curve"][-1]["loss"] <= 1.01 * run["floor"]


def all_four_rollouts(run):
    return all(
        e["passes"][k - 1]["accuracy"] == 1
        for e in run["evaluations"] if e["T"] in (32, 64)
        for k in (e["T"], 2 * e["T"])
    )


assert [r["seed"] for r in runs("window-screen") if within_floor(r)] == [2]
tuned = runs("window-tuning") + runs("window-long-extra")
assert [r["seed"] for r in tuned if within_floor(r)] == [1]
diagnosis = runs("window-round2-diagnosis")
normalized = runs("window-round2-stock-normalized")
screens = {
    "582 parameters": [r for r in diagnosis if r["name"] == "mixed-normalized"] +
                      [r for r in normalized if r["name"] == "normalized-baseline"],
    "270 parameters": [r for r in normalized if r["parameters"] == 270] +
                      runs("window-round2-stock-narrow"),
}
for label, group in screens.items():
    assert len(group) == 5
    assert [r["seed"] for r in group if within_floor(r) and all_four_rollouts(r)] == [2]
    print(label, "seed 2 meets floor and all four rollout endpoints")
for r in runs("window-round2-register"):
    if r["name"] == "gain":
        assert within_floor(r) and all_four_rollouts(r)
for r in runs("window-round2-distribution"):
    if r["name"] == "gain-fixed":
        assert within_floor(r) and all_four_rollouts(r)
for r in runs("window-round2-rule60"):
    for key in ("evaluation", "impulse_evaluation"):
        assert len(r[key]) == 96 and all(p["accuracy"] == 1 for p in r[key])
