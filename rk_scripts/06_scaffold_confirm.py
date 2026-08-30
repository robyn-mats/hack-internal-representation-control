"""Confirm the tag4 scaffold on the cells that actually deviate.

The wording test found tag4 clean at 0/42 where the plain scaffold ran 38%,
including T3 with its ORIGINAL dot filler (100% -> 0%) -- which no filler swap
could achieve. But it covered only 5 of 67 phrasings, and the cells that deviated
most in the full pilot were not among them: C at 22.1% and N at 14.0%.

This runs both scaffolds over the phrasings that matter, using the REAL templates
from irc/conditions.csv with agreement resolved from irc/concepts.csv, so the
frames are exactly what the experiment would use.

Selection, by why each is here:

    high deviation in the pilot   C1 38%, C4 37%, N1 63%, N3, J3, D1, T3, T4
    manipulation check            A1, T1, T7  (A > T1 > T7 must survive)
    clean controls                G1, I5, L1  (must stay at 0%)

Ten pilot concepts, two carriers, both scaffolds.

    python3 rk_scripts/06_scaffold_confirm.py
    python3 rk_scripts/03_run_generation.py --run-id scaffold2 --split pilot \\
        --stimuli scaffold2_stimuli.csv
"""

from irc import env  # noqa: F401  -- must be the first import

import argparse
import csv

from irc.paths import REPO_ROOT

PLAIN = ('Write exactly this sentence and nothing else:\n"{carrier}"\n'
         'While you write it: {frame}.')
PLAIN_NF = 'Write exactly this sentence and nothing else:\n"{carrier}"'
TAG4 = ('The tags below mark a sentence. Output that sentence alone -- no tags, '
        'no commentary.\n<sentence>{carrier}</sentence>\n'
        'While you write it: {frame}.')
TAG4_NF = ('The tags below mark a sentence. Output that sentence alone -- no '
           'tags, no commentary.\n<sentence>{carrier}</sentence>')

PHRASINGS = ["A1", "C1", "C4", "D1", "G1", "I5", "J3", "L1",
             "N1", "N3", "T1", "T3", "T4", "T7"]

PILOT_CONCEPTS = ("amphitheaters", "frosts", "kaleidoscopes", "rubber", "satellites",
                  "secrecy", "silver", "trumpets", "vegetables", "xylophones")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "scaffold2_stimuli.csv")
    ap.add_argument("--carriers", type=int, default=2)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT / "rk_scripts"))
    from screen_carriers import SELECTED_CARRIERS_V1

    conds = {r["phrasing_id"]: r
             for r in csv.DictReader((REPO_ROOT / "irc" / "conditions.csv").open())}
    agree = {r["concept"]: r
             for r in csv.DictReader((REPO_ROOT / "irc" / "concepts.csv").open())}
    missing = [p for p in PHRASINGS if p not in conds]
    if missing:
        raise SystemExit(f"phrasings absent from conditions.csv: {missing}")

    rows, seen = [], set()
    for concept in PILOT_CONCEPTS:
        a = agree[concept]
        for ci, carrier in enumerate(SELECTED_CARRIERS_V1[: args.carriers], start=1):
            for pid in PHRASINGS:
                t = conds[pid]["template"]
                frame = (t.replace("{concept}", concept)
                          .replace("{BE}", a["BE"]).replace("{DO}", a["DO"])
                          .replace("{BELONG}", a["BELONG"]))
                for sname, tmpl, nf in (("plain", PLAIN, PLAIN_NF),
                                        ("tag4", TAG4, TAG4_NF)):
                    prompt = (tmpl.format(carrier=carrier, frame=frame) if frame
                              else nf.format(carrier=carrier))
                    key = f"{sname}_{pid}|{concept if frame else '-'}|{ci}"
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "stimulus_id": key.replace("|", "_"), "prompt_group": key,
                        "phrasing_id": f"{sname}_{pid}",
                        "cell_id": f"{sname}_{conds[pid]['cell_id']}",
                        "family": conds[pid]["family"],
                        "direction": conds[pid]["direction"],
                        "frame_type": conds[pid]["frame"],
                        "negation": conds[pid]["negation"],
                        "concept": concept, "concept_number": a["number"],
                        "split": "pilot", "other_concept": "", "carrier": carrier,
                        "carrier_order": ci, "target": carrier,
                        "similarity_to_this_concept": "", "max_similarity": "",
                        "n_prompt_tokens": "", "prompt": prompt,
                    })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}")
    print(f"  {len(rows)} trials = 2 scaffolds x {len(PHRASINGS)} phrasings x "
          f"{len(PILOT_CONCEPTS)} concepts x {args.carriers} carriers (T7 deduped)")
    print(f"  ~{len(rows) * 1.9 / 60:.0f} min")
    r = next(r for r in rows if r["phrasing_id"] == "tag4_C1")
    print("\n  example (tag4, C1):")
    for line in r["prompt"].splitlines():
        print(f"    {line}")


if __name__ == "__main__":
    main()
