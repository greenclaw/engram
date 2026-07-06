"""Decision-3 auto-gate: confidence-based auto-approve + the μ−Zσ calibration math."""
import json

import pytest

from bench_curate import calibrate, score_changeset
from engram.curate import auto_approvable


# --- auto_approvable ---------------------------------------------------------

def _cs(*changes):
    return {"changes": list(changes)}


def test_all_high_confidence_auto_approves():
    cs = _cs({"op": "ADD", "name": "x", "confidence": 0.95},
             {"op": "INVALIDATE", "target": "y", "invalidated_by": "x", "confidence": 0.9})
    assert auto_approvable(cs, 0.85)


def test_one_below_threshold_blocks():
    cs = _cs({"op": "ADD", "name": "x", "confidence": 0.95},
             {"op": "ADD", "name": "z", "confidence": 0.5})
    assert not auto_approvable(cs, 0.85)


def test_missing_confidence_blocks():
    assert not auto_approvable(_cs({"op": "ADD", "name": "x"}), 0.5)


def test_bool_confidence_is_not_a_number():
    assert not auto_approvable(_cs({"op": "ADD", "name": "x", "confidence": True}), 0.5)


def test_body_update_never_auto_applies():
    # decision 3 carve-out: re-touches of curated prose stay human-gated regardless of confidence
    cs = _cs({"op": "UPDATE", "target": "x", "body": "rewritten", "confidence": 1.0})
    assert not auto_approvable(cs, 0.5)


def test_frontmatter_only_update_may_auto_apply():
    cs = _cs({"op": "UPDATE", "target": "x", "description": "refreshed", "confidence": 0.99})
    assert auto_approvable(cs, 0.9)


def test_noop_only_changeset_does_not_auto_approve():
    assert not auto_approvable(_cs({"op": "NOOP", "confidence": 1.0}), 0.5)


# --- CLI wiring --------------------------------------------------------------

def _store(tmp_path):
    mem = tmp_path / "mem"
    mem.mkdir()
    return mem


def test_cli_auto_threshold_applies_without_prompt(tmp_path, capsys):
    from engram.cli import main

    mem = _store(tmp_path)
    cs = tmp_path / "cs.json"
    cs.write_text(json.dumps({"changes": [{"op": "ADD", "name": "auto-note", "type": "gotcha",
                                           "description": "d", "body": "b", "confidence": 0.97}]}))
    assert main(["curate", "apply", str(cs), "--dir", str(mem), "--auto-threshold", "0.9"]) == 0
    assert (mem / "auto-note.md").exists()
    assert "auto-approved" in capsys.readouterr().out


def test_cli_auto_threshold_falls_back_to_gate_when_low(tmp_path, capsys):
    from engram.cli import main

    mem = _store(tmp_path)
    cs = tmp_path / "cs.json"
    cs.write_text(json.dumps({"changes": [{"op": "ADD", "name": "auto-note", "type": "gotcha",
                                           "description": "d", "body": "b", "confidence": 0.3}]}))
    # non-interactive + below threshold -> gate declines, nothing written
    assert main(["curate", "apply", str(cs), "--dir", str(mem), "--auto-threshold", "0.9"]) == 0
    assert not (mem / "auto-note.md").exists()


# --- calibration -------------------------------------------------------------

def test_calibrate_mu_minus_z_sigma():
    samples = [(0.9, True), (0.8, True), (1.0, True), (0.7, False), (0.95, False)]
    r = calibrate(samples, z=1.0)
    mu, sd = 0.9, (0.02 / 3) ** 0.5  # mean/std of correct confidences
    assert r["threshold"] == pytest.approx(mu - sd)
    assert r["coverage"] == pytest.approx(2 / 3)   # correct decisions at/above tau (0.9, 1.0)
    assert r["risk"] == pytest.approx(1 / 2)       # incorrect decisions at/above tau (0.95)


def test_calibrate_no_correct_samples():
    r = calibrate([(0.9, False)], z=2.0)
    assert r["threshold"] is None


def test_score_changeset_emits_confidence_samples():
    labels = [{"op": "ADD"}, {"op": "NOOP", "target": "t"}]
    changes = [{"candidate": 0, "op": "ADD", "confidence": 0.9},
               {"candidate": 1, "op": "UPDATE", "target": "t", "confidence": 0.6}]
    r = score_changeset(labels, changes)
    assert r["samples"] == [(0.9, True), (0.6, False)]
