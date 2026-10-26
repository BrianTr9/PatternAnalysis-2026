"""
analysis.py - ROUGE of finished runs on extra subsets of the validation split, including the
rows whose report also occurs in the training split (no GPU, no model).

Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662)
Course : COMP3710 Pattern Recognition, UQ, Semester 2 2026 - Project 2.6
         (FLAN-T5 lay summarisation of radiology reports, LoRA vs full fine-tuning)

Description
    train.py reports ROUGE on three tiers of the validation split (full, unseen_report, clean;
    see dataset.py). This script adds the complementary subsets so that the effect of the
    train/validation overlap can be read directly from the saved predictions of every run
    (`val_predictions.jsonl` holds the ROUGE of each validation row):
        seen_report              the report text also occurs in train (the part removed by
                                 `unseen_report`)
        seen_same_summary        ... and the lay summary is also one of the train summaries
                                 of that report (a model that memorised train can copy it)
        seen_different_summary   ... but the lay summary differs from every train summary
        unseen_report_same_study the report is new but its study/session is in train
                                 (in `unseen_report`, excluded from `clean`)
    Comparing these rows with the full and clean tiers separates memorisation from
    summarisation. Nothing is generated here: only the per-row scores are averaged.

Usage
    python analysis.py --runs runs/zeroshot runs/lora_r8_lr0.003 runs/full_lr0.0003
    python analysis.py --self_check      # needs the dataset; checks the subset sizes

References
    Same normalisation and tiers as dataset.py (normalise_report, evaluation_tiers); sizes agree
    with the data audit (4,506 seen rows, of which 2,380 share a train summary).

AI assistance: drafted with Claude (Anthropic) and reviewed by the author, see the README
section "Artificial Intelligence Usage Disclosure".
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

from dataset import (SOURCE_COLUMN, TARGET_COLUMN, evaluation_tiers, load_laysumm,
                     normalise_report)
from utils import ROUGE_TYPES, mean_rouge

TIER_ORDER = ("full", "unseen_report", "clean", "seen_report", "seen_same_summary",
              "seen_different_summary", "unseen_report_same_study")


def validation_subsets(splits) -> Dict[str, List[int]]:
    """Validation row indices of the three tiers of dataset.py plus the four extra subsets."""
    train, validation = splits["train"], splits["validation"]
    summaries_of = defaultdict(set)         # normalised report -> normalised train summaries
    for report, summary in zip(list(train[SOURCE_COLUMN]), list(train[TARGET_COLUMN])):
        summaries_of[normalise_report(report)].add(normalise_report(summary))
    tiers = evaluation_tiers(train, validation)
    unseen, clean = set(tiers["unseen_report"]), set(tiers["clean"])
    reports = [normalise_report(r) for r in list(validation[SOURCE_COLUMN])]
    summaries = [normalise_report(t) for t in list(validation[TARGET_COLUMN])]
    seen = [i for i in range(len(reports)) if i not in unseen]
    same = [i for i in seen if summaries[i] in summaries_of[reports[i]]]
    same_set = set(same)
    tiers.update(seen_report=seen, seen_same_summary=same,
                 seen_different_summary=[i for i in seen if i not in same_set],
                 unseen_report_same_study=[i for i in sorted(unseen) if i not in clean])
    return tiers


def run_table(name: str, rows: List[Dict[str, float]], subsets: Dict[str, List[int]]) -> List[str]:
    """Markdown rows (one per subset) with the mean ROUGE x100 of one run."""
    lines = []
    for tier in TIER_ORDER:
        idx = [i for i in subsets[tier] if i < len(rows)]    # a run may cover only the first rows
        if idx:
            r = mean_rouge(rows, idx)
            lines.append(f"| {name} | {tier} | {len(idx)} | "
                         + " | ".join(f"{r[k]:.2f}" for k in ROUGE_TYPES) + " |")
    return lines


def _self_check() -> None:
    """Subset sizes on the real validation split, and the averaging on synthetic rows."""
    sizes = {k: len(v) for k, v in validation_subsets(load_laysumm()).items()}
    print("subset sizes:", sizes)
    expected = {"full": 10000, "unseen_report": 5494, "clean": 4037, "seen_report": 4506,
                "seen_same_summary": 2380, "seen_different_summary": 2126,
                "unseen_report_same_study": 1457}
    assert sizes == expected, sizes
    rows = [dict.fromkeys(ROUGE_TYPES, 1.0), dict.fromkeys(ROUGE_TYPES, 0.0)]
    table = run_table("x", rows, {k: [0, 1] if k == "full" else [0] for k in TIER_ORDER})
    assert table[0].startswith("| x | full | 2 | 50.00") and "| 1 | 100.00" in table[1], table
    print("analysis self-check passed")


def main(argv=None) -> None:
    """Print (and save) the subset table for the given run folders."""
    p = argparse.ArgumentParser(description="ROUGE of finished runs on validation subsets")
    p.add_argument("--runs", nargs="+", help="run folders containing val_predictions.jsonl")
    p.add_argument("--output", default=os.path.join("runs", "subset_analysis.md"))
    p.add_argument("--self_check", action="store_true")
    args = p.parse_args(argv)
    if args.self_check:
        return _self_check()
    if not args.runs:
        p.error("--runs is required")
    subsets = validation_subsets(load_laysumm())
    lines = ["| run | subset | n | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |",
             "|---|---|---|---|---|---|---|"]
    for run in args.runs:
        with open(os.path.join(run, "val_predictions.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        lines += run_table(os.path.basename(os.path.normpath(run)), rows, subsets)
    print("\n".join(lines))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
