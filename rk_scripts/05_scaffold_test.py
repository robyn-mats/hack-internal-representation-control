"""Test an explicit-delimiter scaffold against the current one.

The filler test settled half the T3/T4 problem. With `blah` as the filler, T4
goes 93% -> 0% deviation and T5 stays at 0%, but T3 stays at 50-71% under every
filler. T4 and T5 both contain `not`; T3 is pure filler plus a noun with no
instruction-like element at all, so there is no word to swap in and no filler
can fix it. The failures show the model narrating rather than copying:

    (While I wrote it: blah blah blah blah blah amphitheaters

-- it paraphrases the instruction into the past tense, which is what a model
does when it cannot find anything actionable in a line.

That is a copy-BOUNDARY failure: where does the text to reproduce end and the
instruction begin? Explicit delimiters address it directly, which no filler swap
can. Testing that here before amending anything.

Both scaffolds keep `While you write it: {frame}.` as the final line, so the
distance from the concept tokens to the response tokens is unchanged and the
comparison is not confounded by position.

Also carries A1, C1, N1 and T7 so the change can be checked for side effects:
the ordering A > T1 > T7 must survive any scaffold change (the pre-registered
manipulation check), and C (22%) and N (14%) are the other high-deviation cells,
where explicit delimiters might help or might introduce tag-echoing.

    python3 rk_scripts/05_scaffold_test.py
    python3 rk_scripts/03_run_generation.py --run-id scaffold1 --split pilot \
        --stimuli scaffold_stimuli.csv --verbose
"""

from irc import env  # noqa: F401  -- must be the first import

import argparse
import csv

from irc.paths import REPO_ROOT

SCAFFOLDS = {
    "plain": ('Write exactly this sentence and nothing else:\n"{carrier}"\n'
              'While you write it: {frame}.'),
    "tagged": ('Write exactly the sentence inside the tags below, and nothing '
               'else:\n<sentence>{carrier}</sentence>\n'
               'While you write it: {frame}.'),
}
SCAFFOLDS_NO_FRAME = {
    "plain": 'Write exactly this sentence and nothing else:\n"{carrier}"',
    "tagged": ('Write exactly the sentence inside the tags below, and nothing '
               'else:\n<sentence>{carrier}</sentence>'),
}

# The cells that matter: the broken one, its controls, the manipulation check,
# and the two other high-deviation cells.
FRAMES = {
    "T1":      "{concept}",
    "T3dot":   ". . . . . {concept}",
    "T3blah":  "blah blah blah blah blah {concept}",
    "T4blah":  "blah blah blah blah not {concept}",
    "T5blah":  "not blah blah blah blah {concept}",
    "A1":      "concentrate on {concept}",
    "C1":      "do not ignore {concept}",
    "N1":      "juggle {concept}",
    "T7":      "",
}

PILOT_CONCEPTS = ("amphitheaters", "frosts", "kaleidoscopes", "rubber", "satellites")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=REPO_ROOT / "scaffold_stimuli.csv")
    ap.add_argument("--carriers", type=int, default=2)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT / "rk_scripts"))
    from screen_carriers import SELECTED_CARRIERS_V1

    rows = []
    for concept in PILOT_CONCEPTS:
        for ci, carrier in enumerate(SELECTED_CARRIERS_V1[: args.carriers], start=1):
            for sname, tmpl in SCAFFOLDS.items():
                for fname, frame_t in FRAMES.items():
                    frame = frame_t.format(concept=concept)
                    prompt = (tmpl.format(carrier=carrier, frame=frame) if frame
                              else SCAFFOLDS_NO_FRAME[sname].format(carrier=carrier))
                    pid = f"{sname}_{fname}"
                    rows.append({
                        "stimulus_id": f"{pid}_{concept}_c{ci}",
                        "prompt_group": f"{pid}|{concept if frame else '-'}|{ci}",
                        "phrasing_id": pid, "cell_id": f"scaffold_{sname}",
                        "family": "scaffold_test", "direction": "neutral",
                        "frame_type": "none", "negation": "none",
                        "concept": concept, "concept_number": "", "split": "pilot",
                        "other_concept": "", "carrier": carrier,
                        "carrier_order": ci, "target": carrier,
                        "similarity_to_this_concept": "", "max_similarity": "",
                        "n_prompt_tokens": "", "prompt": prompt,
                    })

    seen, uniq = set(), []
    for r in rows:                      # T7 names no concept -> one prompt per carrier
        if r["prompt_group"] in seen:
            continue
        seen.add(r["prompt_group"]); uniq.append(r)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(uniq[0].keys()))
        w.writeheader(); w.writerows(uniq)
    print(f"wrote {args.out}\n  {len(uniq)} trials, ~{len(uniq) * 1.9 / 60:.0f} min")
    for s in SCAFFOLDS:
        r = next(r for r in uniq if r["phrasing_id"] == f"{s}_T3blah")
        print(f"\n  --- {s} ---")
        for line in r["prompt"].splitlines():
            print(f"    {line}")


if __name__ == "__main__":
    main()
