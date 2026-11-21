"""Read-only audit of label-window overlap across temporal splits.

The model label is "unscheduled removal within the next 30 flight cycles".
A snapshot near a split boundary can therefore be labelled by an event that
belongs to the next split.  This script quantifies that overlap using the
saved audit links and split files.  It never loads features, scores, models,
or temporal-test metrics, and it never retrains or re-evaluates anything.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINKS = ROOT / "data" / "audit" / "snapshot_event_links.csv"
DEFAULT_SPLITS = ROOT / "data" / "splits" / "temporal"
DEFAULT_OUTPUT = ROOT / "reports" / "label_boundary_audit.json"
SPLITS = ("train", "val", "test")


def audit(links_path: Path, splits_dir: Path) -> dict:
    links = pd.read_csv(links_path)
    window_links = links[links["cycles_before_event"] <= 30].copy()

    same = window_links["same_temporal_split"].astype(bool)
    cross = window_links[~same]

    pairs = (
        cross.groupby(["snapshot_temporal_split", "event_temporal_split"])
        .size()
        .reset_index(name="link_count")
        .sort_values("link_count", ascending=False)
    )

    result = {
        "audit": "temporal_label_window_overlap",
        "read_only": True,
        "temporal_test_metrics_read": False,
        "models_retrained": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "links": str(links_path.relative_to(ROOT)),
            "splits_dir": str(splits_dir.relative_to(ROOT)),
        },
        "links": {
            "total": int(len(links)),
            "within_30_cycles": int(len(window_links)),
            "same_split": int(same.sum()),
            "cross_split": int((~same).sum()),
            "cross_split_pairs": {
                f"{row.snapshot_temporal_split}->{row.event_temporal_split}": int(
                    row.link_count
                )
                for row in pairs.itertuples(index=False)
            },
        },
        "positive_snapshots": {},
        "notes": [
            "Only same-split links are eligible for event recall, so the event "
            "denominator itself is not contaminated.",
            "Cross-split links mean a small share of training/validation targets "
            "are driven by events in the next split (label overlap, not feature leakage).",
            "The frozen temporal test has already been consumed; this audit does "
            "not change the official model, threshold, or reported test result.",
        ],
    }

    for split in SPLITS:
        split_file = splits_dir / f"{split}.csv"
        frame = pd.read_csv(split_file, usecols=["record_id", "target"])
        positives = frame.loc[frame["target"] == 1, "record_id"]
        same_ids = set(
            window_links.loc[
                (window_links["snapshot_temporal_split"] == split) & same, "record_id"
            ]
        )
        cross_ids = set(
            window_links.loc[
                (window_links["snapshot_temporal_split"] == split) & ~same, "record_id"
            ]
        )
        same_linked = int(positives.isin(same_ids).sum())
        cross_only = int((positives.isin(cross_ids) & ~positives.isin(same_ids)).sum())
        result["positive_snapshots"][split] = {
            "total": int(len(positives)),
            "same_split_linked": same_linked,
            "cross_split_only": cross_only,
            "cross_split_any": int(positives.isin(cross_ids).sum()),
            "cross_split_only_rate": round(cross_only / len(positives), 6)
            if len(positives)
            else 0.0,
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = audit(args.links, args.splits_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    links = result["links"]
    print(
        f"links total={links['total']} within_30={links['within_30_cycles']} "
        f"same={links['same_split']} cross={links['cross_split']}"
    )
    for split, row in result["positive_snapshots"].items():
        print(
            f"{split}: positives={row['total']} same_split={row['same_split_linked']} "
            f"cross_only={row['cross_split_only']} ({row['cross_split_only_rate']:.4%})"
        )
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
