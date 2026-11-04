"""Check RoBERTa trainable-parameter counts against Table 1 of the proposal.

Run from the repo root:
    python scripts/param_audit_roberta.py            # downloads roberta-base
    python scripts/param_audit_roberta.py --offline  # same architecture, random weights

Writes results/param_audit_roberta.json and exits with an error if any count is off.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lora_budget.models.roberta import build_model, count_params

# Table 1 of the proposal
EXPECTED_LORA = {4: 147_456, 8: 294_912, 16: 589_824}
EXPECTED_HEAD = 649_805


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="random-init roberta-base architecture")
    ap.add_argument("--out", default="results/param_audit_roberta.json")
    args = ap.parse_args()

    rows, ok = [], True
    for method, rank in [("full_ft", None), ("head_only", None), ("lora", 4), ("lora", 8), ("lora", 16)]:
        kwargs = {"rank": rank, "alpha": 2 * rank} if rank else {}
        model = build_model(method, seed=42, offline_random_init=args.offline, **kwargs)
        c = count_params(model)
        row = {"method": method, "rank": rank, **c}
        if method == "lora":
            row["lora_matches_table1"] = c["lora"] == EXPECTED_LORA[rank]
            ok &= row["lora_matches_table1"]
        if method in ("lora", "head_only"):
            row["head_matches_table1"] = c["head"] == EXPECTED_HEAD
            ok &= row["head_matches_table1"] and c["other_trainable"] == 0
        rows.append(row)
        print(row)
        del model

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"all_match": ok, "rows": rows}, indent=2))
    print("ALL COUNTS MATCH TABLE 1" if ok else "MISMATCH: check the rows above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
