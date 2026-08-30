"""Generate `stimuli.csv` -- the full crossing of conditions x concepts x carriers.

Model-free and CPU-only: it loads a tokenizer to count prompt tokens (a
pre-registered covariate for the dilution check), never the weights.

Inputs, all authoritative and versioned:
    irc/conditions.csv          67 phrasings across 25 cells, with factor coding
    irc/concepts.csv            50 concepts, grammatical number and agreement
    screen_carriers.py          SELECTED_CARRIERS_V1, the 7 screened carriers
    carrier_similarity.csv      optional; supplies the similarity covariates

Output:
    stimuli.csv                 one row per (phrasing, concept, carrier)
    stimuli_meta.json           provenance for the above

One row is one MEASUREMENT. Rows sharing a `prompt_group` share a prompt and
need only one generation: T7 names no concept, so its prompt depends on the
carrier alone and is read out against every concept, exactly as upstream's
`no_mention` baseline is shared across words.

Usage:
    python3 rk_scripts/00_generate_stimuli.py
    python3 rk_scripts/00_generate_stimuli.py --no-tokenize --out /tmp/s.csv
"""

from irc import env  # noqa: F401  -- must be the first import

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from irc.paths import REPO_ROOT
from screen_carriers import SELECTED_CARRIERS_V1

# The prompt scaffold. A COLON, not a comma: a comma leaves the bare-noun
# conditions (T1 "dust", T2 "not dust") ungrammatical while the imperatives stay
# fine, so grammaticality would otherwise vary with condition. The temporal
# anchor ("While you write it") lives here rather than in the templates, so it
# cannot correlate with the focus condition.
SCAFFOLD = 'Write exactly this sentence and nothing else:\n"{carrier}"\nWhile you write it: {frame}.'
SCAFFOLD_NO_FRAME = 'Write exactly this sentence and nothing else:\n"{carrier}"'

# Concept split, drawn once and frozen (PREREGISTRATION.md). Hard-coded rather
# than re-derived so it cannot drift with a library's RNG; verified below to
# partition irc/concepts.csv exactly.
PILOT_CONCEPTS = (
    "amphitheaters", "frosts", "kaleidoscopes", "rubber", "satellites",
    "secrecy", "silver", "trumpets", "vegetables", "xylophones",
)

# Seeded once for the T6 floor-control partner assignment. Same seed as the
# concept split draw.
SPLIT_SEED = 20260827


def load_conditions(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    if len(rows) != 67:
        raise SystemExit(f"{path}: expected 67 phrasings, found {len(rows)}")
    return rows


def load_concepts(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    if len(rows) != 50:
        raise SystemExit(f"{path}: expected 50 concepts, found {len(rows)}")
    return rows


def assign_partners(concepts: list[str], seed: int) -> dict[str, str]:
    """A seeded derangement: every concept gets a partner, none its own.

    T6 asks whether declaring some OTHER concept irrelevant moves the target's
    readout. It is the floor, so the partner must never be the target itself.
    Rejection-sampling a shuffle is the simplest correct derangement here.
    """
    rng = random.Random(seed)
    shuffled = list(concepts)
    for _ in range(10_000):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(concepts, shuffled)):
            return dict(zip(concepts, shuffled))
    raise SystemExit("could not find a derangement for the T6 partners")


def load_similarity(path: Path):
    """-> {(carrier, concept): z}, {carrier: max_z}. Empty if the file is absent."""
    if not path.exists():
        print(f"note: {path.name} not found -- similarity covariates left blank.\n"
              f"      Generate it with screen_carriers.write_carrier_similarity().")
        return {}, {}
    rows = list(csv.reader(path.open()))
    header, body = rows[0][1:], rows[1:]
    per_pair, per_carrier = {}, {}
    for row in body:
        carrier, vals = row[0], [float(v) for v in row[1:]]
        for concept, v in zip(header, vals):
            per_pair[(carrier, concept.lower())] = v
        per_carrier[carrier] = max(vals)
    return per_pair, per_carrier


def render(template: str, concept: str, agree: dict,
           partner: str, partner_agree: dict) -> str:
    """Fill a condition template. Concepts are lowercased into prompts.

    Agreement follows the template's grammatical SUBJECT, which is not always
    the target concept: T6 ("{other_concept} {BE} irrelevant to this task") is
    about the partner, so a mass target paired with a plural partner would
    otherwise render "fountains is irrelevant to this task". T6 is the floor
    control for Q9, so an agreement error there is not cosmetic.
    """
    subject = partner_agree if "{other_concept}" in template else agree
    return (template
            .replace("{concept}", concept)
            .replace("{other_concept}", partner)
            .replace("{BE}", subject["BE"])
            .replace("{DO}", subject["DO"])
            .replace("{BELONG}", subject["BELONG"]))


def build(conditions, concepts, carriers, partners, sim_pair, sim_max):
    by_name = {r["concept"].lower(): r for r in concepts}
    rows = []
    for concept_row in concepts:
        concept = concept_row["concept"].lower()
        split = "pilot" if concept in PILOT_CONCEPTS else "held_out"
        for ci, carrier in enumerate(carriers, start=1):
            for cond in conditions:
                partner = partners[concept]
                frame = render(cond["template"], concept, concept_row,
                               partner, by_name[partner])
                # T7 is the only condition with no third line at all, so its
                # prompt names no concept and is shared across all 50 of them.
                if frame:
                    prompt = SCAFFOLD.format(carrier=carrier, frame=frame)
                    group = f"{cond['phrasing_id']}|{concept}|{ci}"
                else:
                    prompt = SCAFFOLD_NO_FRAME.format(carrier=carrier)
                    group = f"{cond['phrasing_id']}|-|{ci}"
                rows.append({
                    "stimulus_id": f"{cond['phrasing_id']}_{concept}_c{ci}",
                    "prompt_group": group,
                    "phrasing_id": cond["phrasing_id"],
                    "cell_id": cond["cell_id"],
                    "family": cond["family"],
                    "direction": cond["direction"],
                    "frame_type": cond["frame"],
                    "negation": cond["negation"],
                    "concept": concept,
                    "concept_number": concept_row["number"],
                    "split": split,
                    "other_concept": partner if "{other_concept}" in cond["template"] else "",
                    "carrier": carrier,
                    "carrier_order": ci,
                    "target": carrier,
                    "similarity_to_this_concept": sim_pair.get((carrier, concept), ""),
                    "max_similarity": sim_max.get(carrier, ""),
                    "n_prompt_tokens": "",
                    "prompt": prompt,
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "stimuli.csv")
    ap.add_argument("--no-tokenize", action="store_true",
                    help="skip n_prompt_tokens (avoids loading a tokenizer)")
    args = ap.parse_args()

    conditions = load_conditions(REPO_ROOT / "irc" / "conditions.csv")
    concept_rows = load_concepts(REPO_ROOT / "irc" / "concepts.csv")
    carriers = list(SELECTED_CARRIERS_V1)

    names = [r["concept"].lower() for r in concept_rows]
    unknown = set(PILOT_CONCEPTS) - set(names)
    if unknown:
        raise SystemExit(f"pilot concepts absent from concepts.csv: {sorted(unknown)}")

    partners = assign_partners(names, SPLIT_SEED)
    sim_pair, sim_max = load_similarity(REPO_ROOT / "carrier_similarity.csv")
    rows = build(conditions, concept_rows, carriers, partners, sim_pair, sim_max)

    if not args.no_tokenize:
        from irc.model import load_tokenizer
        from irc.model import MODEL_ID
        tok = load_tokenizer(MODEL_ID)
        for r in rows:
            r["n_prompt_tokens"] = len(tok(r["prompt"])["input_ids"])

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_prompts = len({r["prompt_group"] for r in rows})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    meta = {
        "n_rows": len(rows),
        "n_unique_prompts": n_prompts,
        "n_phrasings": len(conditions),
        "n_cells": len({c["cell_id"] for c in conditions}),
        "n_concepts": len(concept_rows),
        "n_carriers": len(carriers),
        "carriers": carriers,
        "pilot_concepts": list(PILOT_CONCEPTS),
        "split_seed": SPLIT_SEED,
        "scaffold": SCAFFOLD,
        "similarity_covariates": bool(sim_pair),
        "git_commit": commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (args.out.parent / f"{args.out.stem}_meta.json").write_text(
        json.dumps(meta, indent=1))

    by_split = {s: sum(1 for r in rows if r["split"] == s)
                for s in ("pilot", "held_out")}
    print(f"wrote {args.out}")
    print(f"  {len(rows):,} rows = {len(conditions)} phrasings x "
          f"{len(concept_rows)} concepts x {len(carriers)} carriers")
    print(f"  {n_prompts:,} unique prompts to generate "
          f"({len(rows) - n_prompts:,} rows share a T7 prompt)")
    print(f"  pilot {by_split['pilot']:,} | held out {by_split['held_out']:,}")


if __name__ == "__main__":
    main()
