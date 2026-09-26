"""Offline tests for tasks/osha_sir: the shared label parser and slot rule, and official GEPA driven through bpto's
clients (skipped unless `gepa` is installed: pip install -e .[bench])."""
import pytest

from tasks.osha_sir.common import SLOT, parse_label, render

LABELS = ["Falls to lower level", "Falls on same level", "Struck by object or equipment", "Other"]


def test_parse_label_is_lenient_and_prefers_longest():
    assert parse_label("Falls to lower level", LABELS) == "Falls to lower level"
    assert parse_label("Category: struck by object or equipment.", LABELS) == "Struck by object or equipment"
    assert parse_label("It could be Falls on same level.\nFalls to lower level", LABELS) == "Falls to lower level"
    assert parse_label("no idea", LABELS) is None


def test_render_restores_a_dropped_slot():
    assert render("Classify: " + SLOT, "x") == ("Classify: x", False)
    text, restored = render("Classify.", "x")
    assert restored and text.endswith("Narrative: x")


def test_official_gepa_runs_through_bpto_clients():
    pytest.importorskip("gepa")
    from bpto.bo import HashEmbedder
    from tasks.osha_sir.pretest import one_run, summary
    from tasks.osha_sir.qei_sampling import QEISampling

    s = summary(one_run(B=400, seed=0))
    assert s["total_metric_calls"] == s["rows_requested"] and s["rows_billed"] == s["task_calls_billed_by_client"]
    assert s["best_val"] > s["seed_val"]
    smp = QEISampling(4, HashEmbedder(), seed=0)
    one_run(B=600, seed=0, sampling=smp)
    assert all(len(e["parents"]) == 4 for e in smp.log) and any(e["mode"] == "qei" for e in smp.log)
