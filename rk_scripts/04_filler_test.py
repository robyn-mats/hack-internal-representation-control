"""Generate a small stimulus set testing alternative filler tokens for T3/T4/T5.

The generated pilot found T3 (`. . . . . X`) and T4 (`. . . . not X`) deviating
on 90%+ of trials, while T5 (`not . . . . X`) deviates on 0%. The model copies
the entire third line back:

    The train arrived precisely on schedule.
    While you write it: . . . . . satellites.

Hypothesis: a third line beginning with `. . . . .` is not parsed as an
instruction, so it reads as more text to reproduce. T5 starts with a word and is
parsed. If that is right, any filler whose line starts with a word should fix it,
and the fix is to the filler token rather than to the prompt scaffold -- which is
why none of the pre-registered scaffold candidates would have helped.

What the filler has to preserve (PLAN.md section 3):
  - semantically inert, contributing no meaning of its own;
  - T4 and T5 must be token-for-token permutations of each other, so that the
    contrast is arrangement alone;
  - comparable length to T1/T2 so T1-vs-T3 and T2-vs-T4 stay interpretable.

Pilot concepts only. Writes a stimuli CSV the ordinary runner consumes:

    python3 rk_scripts/04_filler_test.py
    python3 rk_scripts/03_run_generation.py --run-id filler1 --split pilot \
        --stimuli filler_stimuli.csv
"""

from irc import env  # noqa: F401  -- must be the first import

import argparse
import csv

from irc.paths import REPO_ROOT

SCAFFOLD = ('Write exactly this sentence and nothing else:\n"{carrier}"\n'
            'While you write it: {frame}.')

# Each filler is a token repeated 5x, matching the current dot count.
FILLERS = {
    "dot":   ".",      # current, the control -- must reproduce ~90% deviation
    "dash":  "-",
    "XXX":   "XXX",    # line starts with a word-like token, as T5 does, but
                       # carries no meaning -- unlike a function word such as
                       # "and", which has real coordinating force
    "blah":  "blah",   # clearly inert, unambiguously not content
    "x":     "x",
    "hmm":   "hmm",
}

# The three arrangements, as in PLAN.md. T4 and T5 are permutations.
ARRANGEMENTS = {
    "T3": "{f} {f} {f} {f} {f} {concept}",
    "T4": "{f} {f} {f} {f} not {concept}",
    "T5": "not {f} {f} {f} {f} {concept}",
}

PILOT_CONCEPTS = ("amphitheaters", "frosts", "kaleidoscopes", "rubber", "satellites")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "filler_stimuli.csv")
    ap.add_argument("--carriers", type=int, default=3)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT / "rk_scripts"))
    from screen_carriers import SELECTED_CARRIERS_V1

    rows = []
    for concept in PILOT_CONCEPTS:
        for ci, carrier in enumerate(SELECTED_CARRIERS_V1[: args.carriers], start=1):
            for fname, tok in FILLERS.items():
                for arr, template in ARRANGEMENTS.items():
                    frame = template.format(f=tok, concept=concept)
                    pid = f"{arr}_{fname}"
                    rows.append({
                        "stimulus_id": f"{pid}_{concept}_c{ci}",
                        "prompt_group": f"{pid}|{concept}|{ci}",
                        "phrasing_id": pid, "cell_id": f"filler_{fname}",
                        "family": "baseline", "direction": "neutral",
                        "frame_type": "none", "negation":
                            "none" if arr == "T3" else "syntactic",
                        "concept": concept, "concept_number": "", "split": "pilot",
                        "other_concept": "", "carrier": carrier,
                        "carrier_order": ci, "target": carrier,
                        "similarity_to_this_concept": "", "max_similarity": "",
                        "n_prompt_tokens": "",
                        "prompt": SCAFFOLD.format(carrier=carrier, frame=frame),
                    })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}")
    print(f"  {len(rows)} trials = {len(FILLERS)} fillers x {len(ARRANGEMENTS)} "
          f"arrangements x {len(PILOT_CONCEPTS)} concepts x {args.carriers} carriers")
    print(f"  ~{len(rows) * 1.9 / 60:.0f} min at the pilot's observed rate")
    print("\n  example frames:")
    for fname in FILLERS:
        r = next(r for r in rows if r["phrasing_id"] == f"T3_{fname}")
        print(f"    {r['prompt'].splitlines()[2]}")


if __name__ == "__main__":
    main()
