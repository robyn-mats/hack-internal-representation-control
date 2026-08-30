"""Run the generation grid over `stimuli.csv` and capture activations.

One record per UNIQUE PROMPT, not per stimulus row. T7 names no concept, so its
prompt depends only on the carrier and is read out against all 50 concepts --
23,450 stimulus rows collapse to 23,107 generations. The measure stage joins
back on `prompt_group`.

Two modes, both pre-registered:

  generate        greedy decoding, the primary method. What the model actually
                  writes differs across conditions, so activation differences
                  are confounded with what was written -- but the alternative is
                  worse (below).
  --teacher-force the target sentence is forced as the assistant turn and the
                  per-token surprisal of the forced tokens is recorded. Forced
                  text is off-distribution ASYMMETRICALLY: if the model deviates
                  on 60% of one cell's trials and 5% of another's, forcing pushes
                  the first further off-distribution and the between-condition
                  difference partly measures surprise at imposed text. Flat
                  surprisal across conditions means forcing is near-neutral;
                  surprisal tracking condition means it is not, and the check has
                  told you the comparison is confounded.

Compliance is an OUTCOME here, not a filter. Upstream captures activations only
`if exact`, discarding ~2/3 of trials -- and, by its own note, possibly the very
trials where the model engaged with the concept hardest. This captures
regardless, and records deviation and leak per trial for per-cell rates.

Leak is scored against ALL 50 concepts rather than just the prompted one. That
costs nothing, makes T7 work (its completion is shared across concepts, so leak
has to be per-concept at measure time), and supplies the second-concept readout
the dilution check needs. Two tiers are recorded -- inflectional (primary) and
inflectional+derivational (sensitivity) -- and both score the COMPLETION only.
The prompt contains the concept by construction, so scoring it would mark every
non-T7 trial as a leak.

Storage: capture defaults to `constants.SAE_LAYERS` (4 layers, ~10 GB for the
full grid). All 62 layers would be 154 GB and the pre-registration restricts the
analysis layer to those four anyway; the other 58 exist only for the layer-curve
secondary figure. Pass --capture-layers all for that, on a subset.

Usage (the model must fit; a 27B bf16 copy needs ~53 GB):
    python3 rk_scripts/03_run_generation.py --run-id pilot1 --split pilot
    python3 rk_scripts/03_run_generation.py --run-id pilot1 --split pilot --teacher-force
    python3 rk_scripts/03_run_generation.py --run-id full1 --split all
    python3 rk_scripts/03_run_generation.py --run-id curve --split pilot \
        --capture-layers all --limit 500

Resumable: re-invoking with the same --run-id skips prompt_groups already in
generations.jsonl. Every invocation is appended to invocations.jsonl.
"""

from irc import env  # noqa: F401  -- must be the first import

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

import torch

from irc.constants import SAE_LAYERS
from irc.model import (
    MODEL_ID,
    ResidualCapture,
    chat_ids,
    get_decoder_layers,
    load_model,
    load_tokenizer,
)
from irc.paths import REPO_ROOT, RUNS


def _pattern(forms: list[str]) -> re.Pattern:
    """Word-boundary alternation, longest-first so `dusting` wins over `dust`.

    Boundaries, not substrings: the pattern for `bags` must not fire on
    "baggage".
    """
    alts = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"\b({alts})\b", re.IGNORECASE)


def load_concept_forms(path: Path) -> tuple[dict[str, re.Pattern], dict[str, re.Pattern]]:
    """-> (strict, loose) leak patterns per concept, from irc/concepts.csv.

    strict  INFLECTIONAL forms only -- the pre-registered leak measure. Catches
            dust/dusts/dusting/dusted, snow/snowed/snowing. Generated forms that
            are not real words ("lightninged") cost nothing, since such strings
            never occur in text.
    loose   strict plus hand-pruned WordNet DERIVATIONAL forms (bloody, snowy,
            rubberize, pacify). Reported as a sensitivity check, never as the
            primary measure: WordNet relates lemmas by string, so polysemous
            concepts import derivations from unrelated senses. The three worst
            (Phones->phonetic, Deserts->desertion, Information->inform) are
            dropped in gen_concepts_csv.py, but the tier stays looser than the
            experiment's claim, which is why it is secondary.

    Neither tier disambiguates sense: a completion using "dust" in an unrelated
    sense still scores as a leak.
    """
    strict, loose = {}, {}
    for row in csv.DictReader(path.open()):
        base = (row.get("forms") or row["concept"]).split("|")
        strict[row["concept"]] = _pattern(base)
        extra = [f for f in (row.get("forms_derived") or "").split("|") if f]
        loose[row["concept"]] = _pattern(base + extra)
    return strict, loose


def detect_leaks(completion: str, patterns: dict[str, re.Pattern]) -> list[str]:
    return [c for c, p in patterns.items() if p.search(completion)]


def load_stimuli(path: Path, split: str) -> list[dict]:
    """One row per unique prompt_group, carrying the fields generation needs.

    Note the splits overlap by exactly 7: T7 names no concept, so its prompt
    depends only on the carrier and is the baseline for pilot AND held-out
    concepts alike. pilot (4,627) + held_out (18,487) = 23,114 against 23,107
    for `all`. That is correct, not double-counting -- running the two splits
    separately regenerates those 7 prompts, which is trivial and keeps each
    run self-contained.
    """
    seen, jobs = set(), []
    for row in csv.DictReader(path.open()):
        if split != "all" and row["split"] != split:
            continue
        if row["prompt_group"] in seen:
            continue
        seen.add(row["prompt_group"])
        jobs.append(row)
    return jobs


def _capture(model, out_ids, lo: int, hi: int, layers: list[int]) -> torch.Tensor:
    """One forward pass, activations for token positions [lo, hi).

    to_cpu=False plus no_grad: the default hook does .float().cpu() per layer,
    which is one synchronising transfer per layer inside a single forward pass,
    and without no_grad the pass builds an autograd graph across every layer.
    Together those made capture 87% of runtime (see NOTES.md).
    """
    with torch.no_grad(), ResidualCapture(model, layers, to_cpu=False) as cap:
        model(out_ids[:, :hi])
    return torch.stack([cap.acts[i][0, lo:hi] for i in layers]).to(torch.bfloat16).cpu().clone()


def generate_one(model, tokenizer, prompt: str, target: str,
                 layers: list[int]) -> dict:
    ids = chat_ids(tokenizer, prompt)
    n_prompt = ids.shape[1]
    tgt_len = len(tokenizer(target, add_special_tokens=False)["input_ids"])
    out = model.generate(ids, max_new_tokens=tgt_len + 16, do_sample=False)

    gen = out[0, n_prompt:]
    end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ends = (gen == end_id).nonzero()
    n_resp = int(ends[0]) if len(ends) else len(gen)
    completion = tokenizer.decode(gen[:n_resp], skip_special_tokens=True).strip()

    acts = _capture(model, out, n_prompt, n_prompt + n_resp, layers)
    return {"completion": completion, "exact_match": completion == target,
            "n_resp_tokens": n_resp, "surprisal": None, "acts": acts}


def teacher_force_one(model, tokenizer, prompt: str, target: str,
                      layers: list[int]) -> dict:
    """Force `target` as the assistant turn; record surprisal of forced tokens."""
    ids = chat_ids(tokenizer, prompt)
    n_prompt = ids.shape[1]
    tgt = torch.tensor(
        [tokenizer(target, add_special_tokens=False)["input_ids"]],
        device=ids.device)
    full = torch.cat([ids, tgt], dim=1)
    n_resp = tgt.shape[1]

    with torch.no_grad():
        logits = model(full).logits
    # logits[t] predicts token t+1, so the forced token at n_prompt+i is scored
    # by the distribution at n_prompt+i-1.
    logprobs = torch.log_softmax(logits[0, n_prompt - 1 : n_prompt + n_resp - 1].float(), -1)
    tok_lp = logprobs.gather(-1, tgt[0].unsqueeze(-1)).squeeze(-1)

    acts = _capture(model, full, n_prompt, n_prompt + n_resp, layers)
    return {"completion": target, "exact_match": True, "n_resp_tokens": n_resp,
            "surprisal": (-tok_lp).tolist(), "acts": acts}


def run(model, tokenizer, run_dir: Path, jobs: list[dict], layers: list[int],
        teacher_force: bool, patterns: dict[str, re.Pattern],
        patterns_loose: dict[str, re.Pattern] | None = None) -> None:
    acts_dir = run_dir / "acts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    gen_path = run_dir / "generations.jsonl"

    done = set()
    if gen_path.exists():
        with gen_path.open() as fh:
            done = {json.loads(line)["prompt_group"] for line in fh}
    todo = [j for j in jobs if j["prompt_group"] not in done]
    print(f"[run] {len(todo)} to do, {len(done)} already present, "
          f"capturing {len(layers)} layers")

    t0, n_dev = time.perf_counter(), 0
    with gen_path.open("a") as fh:
        for i, job in enumerate(todo, 1):
            fn = teacher_force_one if teacher_force else generate_one
            res = fn(model, tokenizer, job["prompt"], job["target"], layers)

            key = job["prompt_group"].replace("|", "__")
            torch.save(res["acts"], acts_dir / f"{key}.pt")
            if not res["exact_match"]:
                n_dev += 1

            fh.write(json.dumps({
                "prompt_group": job["prompt_group"],
                "phrasing_id": job["phrasing_id"], "cell_id": job["cell_id"],
                "concept": job["concept"], "carrier_order": job["carrier_order"],
                "target": job["target"],
                "completion": res["completion"],
                "exact_match": res["exact_match"],
                "n_resp_tokens": res["n_resp_tokens"],
                "leaked_concepts": detect_leaks(res["completion"], patterns),
                "leaked_concepts_loose": detect_leaks(res["completion"], patterns_loose)
                                         if patterns_loose else None,
                "surprisal": res["surprisal"],
                "acts_file": f"acts/{key}.pt",
                "capture_layers": layers,
            }) + "\n")
            fh.flush()

            if i % 100 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)}  {el / i:.2f}s/trial  "
                      f"eta {(len(todo) - i) * el / i / 3600:.1f}h  "
                      f"deviations {n_dev} ({n_dev / i:.0%})")
    print(f"[run] done. {n_dev}/{len(todo)} non-exact this invocation.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="pilot", choices=["pilot", "held_out", "all"])
    ap.add_argument("--capture-layers", default="sae",
                    help="'sae' (default, 4 layers), 'all' (62), or 16,31,40,53")
    ap.add_argument("--teacher-force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--stimuli", type=Path, default=REPO_ROOT / "stimuli.csv")
    args = ap.parse_args()

    tokenizer = load_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID)
    n_layers = len(get_decoder_layers(model))

    if args.capture_layers == "sae":
        layers = list(SAE_LAYERS)
    elif args.capture_layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.capture_layers.split(",")]
    if max(layers) >= n_layers:
        raise SystemExit(f"layer {max(layers)} out of range for {n_layers} layers")

    jobs = load_stimuli(args.stimuli, args.split)
    if args.limit:
        jobs = jobs[: args.limit]

    run_dir = RUNS / args.run_id / ("teacher_forced" if args.teacher_force else "generated")
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    with (run_dir / "invocations.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": commit, "model_id": MODEL_ID,
            "split": args.split, "n_jobs": len(jobs),
            "capture_layers": layers, "teacher_force": args.teacher_force,
            "stimuli": str(args.stimuli), "limit": args.limit,
        }) + "\n")

    strict, loose = load_concept_forms(REPO_ROOT / "irc" / "concepts.csv")
    run(model, tokenizer, run_dir, jobs, layers, args.teacher_force, strict, loose)


if __name__ == "__main__":
    main()
