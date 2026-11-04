"""Tests for the learning-rate selection rule (proposal Section 8.2)."""

import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "select_lr", Path(__file__).resolve().parents[1] / "scripts" / "select_lr_banking77.py")
select_lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select_lr)


def _row(cfg, lr, acc, epoch, phase="lr_search", status="completed", notes="", run=None):
    return {"run_id": run or f"{phase}-{cfg}-{lr}", "phase": phase, "status": status,
            "config": f"configs/banking77/{cfg}.yaml", "method": cfg, "lr": lr,
            "tune_accuracy": acc, "tune_macro_f1": acc - 1, "best_epoch": epoch, "notes": notes}


def test_highest_accuracy_wins():
    df = pd.DataFrame([_row("lora_r8", 1e-4, 90.0, 5), _row("lora_r8", 3e-4, 91.0, 5)])
    assert select_lr.select(df)["lora_r8"]["lr"] == 3e-4


def test_tie_goes_to_earlier_epoch_then_smaller_lr():
    df = pd.DataFrame([_row("full_ft", 1e-5, 90.0, 4), _row("full_ft", 3e-5, 90.0, 2)])
    assert select_lr.select(df)["full_ft"]["lr"] == 3e-5          # earlier epoch wins
    df = pd.DataFrame([_row("full_ft", 3e-5, 90.0, 3), _row("full_ft", 1e-5, 90.0, 3)])
    assert select_lr.select(df)["full_ft"]["lr"] == 1e-5          # then smaller lr


def test_ignores_failed_test_and_other_phase_rows():
    df = pd.DataFrame([
        _row("head_only", 1e-4, 80.0, 5),
        _row("head_only", 3e-4, 99.0, 5, status="failed_code_or_data"),
        _row("head_only", 3e-4, 98.0, 5, notes="TEST RUN (offline) - not evidence.", run="x"),
        _row("head_only", 3e-4, 97.0, 5, phase="confirm", run="y"),
    ])
    assert select_lr.select(df)["head_only"]["lr"] == 1e-4
