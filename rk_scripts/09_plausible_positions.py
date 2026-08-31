"""Pooling rule 3: where was the concept a plausible next token?

PREREGISTRATION.md's third pooling candidate reads the concept out "at positions
where the concept is a plausible next token". That needs next-token
probabilities, which the runner does not store -- it saves activations, not
logits. So this is a separate, model-dependent pass rather than part of the
model-free measure stage.

For each trial, re-run the prompt plus its response through the model once and
record, at every response position, P(the concept's first token | everything so
far). Those become weights: `pool()` in 08_measure.py takes a weighted mean, so
the readout is taken where the concept could actually have surfaced rather than
uniformly across tokens the concept had no chance of appearing at.

Why this rule is worth the extra pass. `token_mean` dilutes -- it averages over
positions where nothing could have happened. A carrier is 7-11 tokens and the
concept plausibly follows only some of them. The other pre-registered candidate,
`topk_mean`, was dropped because it selects positions by the quantity being
measured; this one selects by an independent quantity, which is the whole point.

Concept-free conditions (T7) have no first token to score and are skipped.

    python3 rk_scripts/09_plausible_positions.py --run-id pilot2 --pass generated
"""

from irc import env  # noqa: F401  -- must be the first import

if __name__ == "__main__":
    print("==> 09_plausible_positions: importing torch/transformers (~60s)...",
          flush=True)

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from irc.model import MODEL_ID, chat_ids, load_model, load_tokenizer
from irc.paths import REPO_ROOT, RUNS


@torch.no_grad()
def plausibility(model, tokenizer, prompt: str, target: str,
                 concept: str, n_capture: int) -> list[float]:
    """P(concept's first token | prefix) at each captured response position.

    The concept's FIRST token, not the whole word: a multi-token concept only
    ever begins at a position, and asking for the whole word would require
    marginalising over continuations that the position cannot express.

    Both a leading-space and a bare form are scored and the larger taken --
    mid-sentence the tokenizer would produce " satellites", at a boundary
    "satellites", and which applies varies by position.
    """
    ids = chat_ids(tokenizer, prompt)
    n_prompt = ids.shape[1]
    tgt_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    full = torch.cat([ids, torch.tensor([tgt_ids], device=ids.device)], dim=1)

    cand = set()
    for form in (" " + concept.lower(), concept.lower()):
        t = tokenizer(form, add_special_tokens=False)["input_ids"]
        if t:
            cand.add(t[0])

    logits = model(full).logits
    # logits[t] predicts token t+1, so the distribution *at* response position i
    # is logits[n_prompt + i - 1].
    rows = torch.log_softmax(
        logits[0, n_prompt - 1 : n_prompt + n_capture - 1].double(), -1)
    probs = rows.exp()
    return [float(max(probs[i, c] for c in cand)) for i in range(rows.shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pass", dest="pass_", default="generated",
                    choices=["generated", "teacher_forced"])
    ap.add_argument("--stimuli", type=Path, default=REPO_ROOT / "stimuli.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run_dir = RUNS / args.run_id / args.pass_
    gen_path = run_dir / "generations.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"no run at {gen_path}")

    stimuli = {}
    for r in csv.DictReader(args.stimuli.open()):
        stimuli.setdefault(r["prompt_group"], r)
    records = {}
    with gen_path.open() as fh:
        for line in fh:                      # last record per group wins
            r = json.loads(line)
            records[r["prompt_group"]] = r

    jobs = [(pg, rec) for pg, rec in records.items()
            if pg in stimuli and stimuli[pg]["concept"]]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs):,} trials to score "
          f"({len(records) - len(jobs):,} skipped: concept-free or unmatched)")

    out_path = run_dir / "results" / "plausible_positions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [j for j in jobs if j[0] not in done]
    print(f"  {len(todo):,} to do, {len(done):,} already present")

    tokenizer = load_tokenizer(MODEL_ID)
    print("  loading model...", flush=True)
    model = load_model(MODEL_ID)

    t0 = time.perf_counter()
    for i, (pg, rec) in enumerate(todo, 1):
        st = stimuli[pg]
        done[pg] = plausibility(model, tokenizer, st["prompt"], st["target"],
                                st["concept"], rec["n_capture_tokens"])
        if i % 250 == 0 or i == len(todo):
            out_path.write_text(json.dumps(done))
            el = time.perf_counter() - t0
            print(f"  {i}/{len(todo)}  {el / i:.2f}s/trial  "
                  f"eta {(len(todo) - i) * el / i / 60:.0f} min")
    out_path.write_text(json.dumps(done))

    vals = [v for w in done.values() for v in w]
    print(f"\nwrote {out_path}  ({len(done):,} trials)")
    print(f"  P(concept next): median {sorted(vals)[len(vals) // 2]:.2e}, "
          f"max {max(vals):.3f}")
    print("  08_measure.py picks this up automatically as pooling 'plausible'")


if __name__ == "__main__":
    main()
