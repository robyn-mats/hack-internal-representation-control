"""Build a stimulus file containing only the trials that deviated.

Teacher-forcing the whole grid is nearly pure waste. On a compliant trial the
forced and generated activations are BIT-IDENTICAL -- same prompt, same tokens,
deterministic forward pass, verified on pilot1 at max|diff| = 0. So the forced
pass carries unique activation information only where the model did NOT write
the carrier verbatim.

Under the tagged scaffold that is 2 trials in 4,627 (0.04%), which extrapolates
to roughly 12 of the 18,760 held-out prompts. Teacher-forcing all of them costs
~9.4 h to learn about ~12.

This emits a stimulus file of exactly the non-exact trials from a completed run,
so the forced pass covers every informative case at a cost of seconds:

    python3 rk_scripts/10_deviant_stimuli.py --run-id pilot2
    python3 rk_scripts/03_run_generation.py --run-id pilot2 --split pilot \\
        --teacher-force --stimuli deviant_stimuli.csv

What this trades away: `p_stop_soon`, the graded reluctance-to-stop measure, is
then available only for deviating trials. It appears in no confirmatory contrast
(Q0-Q10), and the pilot's full forced pass already characterises it across every
cell -- which is what makes dropping it for held-out defensible rather than
merely convenient.

Ordering matters: the generated pass must finish first, since which trials
deviated is not knowable until it has run.
"""

from irc import env  # noqa: F401  -- must be the first import

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from irc.paths import REPO_ROOT, RUNS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pass", dest="pass_", default="generated")
    ap.add_argument("--stimuli", type=Path, default=REPO_ROOT / "stimuli.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "deviant_stimuli.csv")
    args = ap.parse_args()

    gen_path = RUNS / args.run_id / args.pass_ / "generations.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"no run at {gen_path}")
    inv = (RUNS / args.run_id / args.pass_ / "invocations.jsonl").read_text()
    if '"event": "completed"' not in inv:
        print("WARNING: that run has no completion record. Trials it never "
              "reached will look compliant by omission, so the filter would be "
              "incomplete. Finish the generated pass first.")

    records = {}
    with gen_path.open() as fh:
        for line in fh:                      # last record per group wins
            r = json.loads(line)
            records[r["prompt_group"]] = r
    deviant = {pg for pg, r in records.items() if not r["exact_match"]}

    # One row per prompt_group. prompt_group is NOT unique in stimuli.csv: each
    # T7 group maps to 50 rows, one per concept, because T7 names no concept and
    # its prompt is shared. Without this, a single deviant T7 trial would emit 50
    # rows and the forced pass would run 50 identical prompts.
    seen, rows = set(), []
    for r in csv.DictReader(args.stimuli.open()):
        if r["prompt_group"] in deviant and r["prompt_group"] not in seen:
            seen.add(r["prompt_group"])
            rows.append(r)
    if not rows:
        print(f"no deviant trials in {args.run_id}/{args.pass_} -- nothing to "
              f"teacher-force, and that is the result, not a failure.")
        return

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"wrote {args.out}")
    print(f"  {len(rows)} rows from {len(deviant)} deviant prompt_groups "
          f"of {len(records):,} ({len(deviant) / len(records):.2%})")
    print(f"  by cell: {dict(Counter(r['cell_id'] for r in rows))}")
    print(f"  saved versus forcing the whole run: "
          f"~{(len(records) - len(rows)) * 1.4 / 3600:.1f} h")


if __name__ == "__main__":
    main()
