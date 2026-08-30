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

# Before the heavy imports, not inside main(): `import torch` plus transformers
# is ~60s on this pod, and main() does not run until they finish. Without this
# the script is silent for a minute before it says anything at all. Guarded so
# importing this module (e.g. from the notebook) stays quiet.
if __name__ == "__main__":
    print("==> 03_run_generation: importing torch/transformers (~60s)...", flush=True)

import argparse
import csv
import datetime
import json
import math
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


def _pattern(words: list[str], emoji: list[str] = ()) -> re.Pattern:
    """Word-boundary alternation for words, literal alternation for emoji.

    Boundaries, not substrings: the pattern for `bags` must not fire on
    "baggage". But `\b` is defined between a word and a non-word character, so
    it never matches around an emoji -- `\b🎺\b` finds nothing. Emoji are
    therefore matched literally, in a separate branch.
    """
    parts = []
    if words:
        alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        parts.append(rf"\b(?:{alts})\b")
    if emoji:
        parts.append("|".join(re.escape(e) for e in emoji))
    return re.compile("|".join(parts), re.IGNORECASE)


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
        emoji = [e for e in (row.get("forms_emoji") or "").split("|") if e]
        # Emoji count as leaks under BOTH tiers: a concept reaching the output
        # pictorially has surfaced just as much as one reaching it lexically,
        # and unlike the derivational forms there is no sense ambiguity.
        strict[row["concept"]] = _pattern(base, emoji)
        extra = [f for f in (row.get("forms_derived") or "").split("|") if f]
        loose[row["concept"]] = _pattern(base + extra, emoji)
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


_WHITESPACE_IDS: dict[int, list[int]] = {}


def _whitespace_ids(tokenizer) -> list[int]:
    """Token ids whose decoded form is pure whitespace, cached per tokenizer."""
    out = []
    for tid in range(len(tokenizer)):
        try:
            t = tokenizer.decode([tid])
        except Exception:
            continue
        if t and t.strip() == "":
            out.append(tid)
    return out


def _capture(model, out_ids, lo: int, hi: int, layers: list[int]) -> torch.Tensor:
    """One forward pass, activations for token positions [lo, hi).

    to_cpu=False plus no_grad: the default hook does .float().cpu() per layer,
    which is one synchronizing transfer per layer inside a single forward pass,
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

    # Capture exactly the carrier's own token span, not everything up to
    # <end_of_turn>. Gemma often appends a trailing "\n" that .strip() removes
    # before the exactness check, so a trial can be exact_match=True and still
    # have written 8 tokens rather than 7. Capturing all of them would average a
    # semantically empty newline into the readout for some trials and not
    # others -- and which trials is condition-dependent, so it is a
    # condition-correlated dilution of the dependent variable. It would also
    # give the two passes different shapes for the same stimulus.
    n_cap = min(n_resp, tgt_len)
    acts = _capture(model, out, n_prompt, n_prompt + n_cap, layers)
    return {"completion": completion, "exact_match": completion == target,
            "n_resp_tokens": n_resp, "n_capture_tokens": n_cap,
            "surprisal": None, "eot_surprisal": None,
            "eot_top_token": None, "eot_top_p": None,
            "p_stop_direct": None, "p_stop_soon": None,
            "ws_path": None, "after_ws_token": None, "acts": acts}


def teacher_force_one(model, tokenizer, prompt: str, target: str,
                      layers: list[int]) -> dict:
    """Force `target` as the assistant turn; record surprisal of forced tokens."""
    ids = chat_ids(tokenizer, prompt)
    n_prompt = ids.shape[1]
    tgt_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    n_resp = len(tgt_ids)

    # Score <end_of_turn> as well as the carrier tokens. Forcing the carrier is
    # unsurprising for a model that was going to write it anyway -- N1 ("juggle
    # X") reproduced the carrier correctly and only then appended a comment, so
    # every forced token was near-zero surprisal while the actual deviation sat
    # one token past the end. Reluctance to STOP is where deviation shows up
    # under forcing, and it is invisible unless the stop token is scored.
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    scored = torch.tensor([tgt_ids + [eot_id]], device=ids.device)
    full = torch.cat([ids, scored], dim=1)

    with torch.no_grad():
        logits = model(full).logits
    # logits[t] predicts token t+1, so the forced token at n_prompt+i is scored
    # by the distribution at n_prompt+i-1. The final row scores <end_of_turn>.
    # float64, not float32. Gemma copies the carrier with near-certainty when the
    # source is in context, so compliant trials saturate: in float32 any p above
    # ~0.99999988 reads as exactly log(p) = 0, and two conditions with genuinely
    # different (tiny) surprisal become indistinguishable. float64 costs nothing
    # here and removes that artificial floor. The real floor is the bf16 logits
    # upstream, so differences far below ~1e-3 nats should not be trusted either
    # way -- see NOTES.md.
    logprobs = torch.log_softmax(
        logits[0, n_prompt - 1 : n_prompt + n_resp].double(), -1)
    tok_lp = logprobs.gather(-1, scored[0].unsqueeze(-1)).squeeze(-1)

    # What the model would rather have written than stopping.
    stop_row = logprobs[-1]
    top_lp, top_id = stop_row.max(-1)

    # Whitespace is a PREFIX to continuation, not an alternative to stopping.
    # A compliant trial writes "\n" then stops; a deviating one writes " " then
    # keeps going. Both put their mass on whitespace at this position, so one
    # token of lookahead cannot separate them -- and simply excluding whitespace
    # scores the genuinely-continuing trial at zero, which is backwards.
    #
    # Walk the greedy path forward while it stays whitespace (at most 3 steps,
    # since Gemma's habit is one or two), accumulating the probability of
    # stopping along the way. p_stop_soon is then "would this have ended the
    # turn", which is the question; p_stop_direct is the immediate P(stop).
    ws_ids = set(_WHITESPACE_IDS.setdefault(id(tokenizer), _whitespace_ids(tokenizer)))
    probs = stop_row.exp()
    p_stop_direct = float(probs[eot_id])

    p_stop_soon = p_stop_direct
    p_path, path, after = 1.0, [], None
    cur = torch.cat([ids, torch.tensor([tgt_ids], device=ids.device)], dim=1)
    row = probs
    for _ in range(3):
        top_id = int(row.argmax())
        if top_id not in ws_ids:
            after = tokenizer.decode([top_id])
            break
        p_path *= float(row[top_id])
        path.append(tokenizer.decode([top_id]))
        cur = torch.cat([cur, torch.tensor([[top_id]], device=ids.device)], dim=1)
        with torch.no_grad():
            row = torch.log_softmax(model(cur).logits[0, -1].double(), -1).exp()
        p_stop_soon += p_path * float(row[eot_id])
    else:
        after = tokenizer.decode([int(row.argmax())])

    # Capture excludes the stop token, matching the generated pass exactly.
    acts = _capture(model, full, n_prompt, n_prompt + n_resp, layers)
    return {"completion": target, "exact_match": True, "n_resp_tokens": n_resp,
            "surprisal": (-tok_lp[:n_resp]).tolist(),
            "eot_surprisal": float(-tok_lp[n_resp]),
            "eot_top_token": tokenizer.decode([int(top_id)]),
            "eot_top_p": float(top_lp.exp()),
            "p_stop_direct": p_stop_direct, "p_stop_soon": min(1.0, p_stop_soon),
            "ws_path": path, "after_ws_token": after,
            "n_capture_tokens": n_resp, "acts": acts}


def run(model, tokenizer, run_dir: Path, jobs: list[dict], layers: list[int],
        teacher_force: bool, patterns: dict[str, re.Pattern],
        patterns_loose: dict[str, re.Pattern] | None = None,
        verbose: bool = False) -> None:
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
                "n_capture_tokens": res["n_capture_tokens"],
                "p_stop_direct": res["p_stop_direct"],
                "p_stop_soon": res["p_stop_soon"],
                "ws_path": res["ws_path"],
                "after_ws_token": res["after_ws_token"],
                "leaked_concepts": detect_leaks(res["completion"], patterns),
                "leaked_concepts_loose": detect_leaks(res["completion"], patterns_loose)
                                         if patterns_loose else None,
                "surprisal": res["surprisal"],
                "eot_surprisal": res["eot_surprisal"],
                "eot_top_token": res["eot_top_token"],
                "eot_top_p": res["eot_top_p"],
                "acts_file": f"acts/{key}.pt",
                "capture_layers": layers,
            }) + "\n")
            fh.flush()

            if verbose:
                # The scaffold is constant; the frame (third line) is what
                # varies, so print that rather than repeating the boilerplate.
                lines = job["prompt"].splitlines()
                frame = lines[2] if len(lines) > 2 else "(no third line -- T7)"
                mark = "EXACT" if res["exact_match"] else "*** NOT EXACT ***"
                leaks = detect_leaks(res["completion"], patterns)
                print(f"\n[{i}/{len(todo)}] {job['phrasing_id']} {job['cell_id']}"
                      f"  {job['concept'] or '-'}/c{job['carrier_order']}")
                print(f"   frame: {frame}")
                print(f"   got:   {res['completion']!r}  {mark}")
                if leaks:
                    print(f"   LEAK:  {leaks}")
                if res["surprisal"]:
                    sp = res["surprisal"]
                    print(f"   surprisal: mean {sum(sp) / len(sp):.4f} "
                          f"max {max(sp):.4f} nats over {len(sp)} forced tokens")
                    pstop = math.exp(-res["eot_surprisal"])
                    tok = res["eot_top_token"]
                    pref = ("stop is the top choice" if tok == "<end_of_turn>"
                            else f"prefers {tok!r} (p={res['eot_top_p']:.3f})")
                    print(f"   stop:      P(stop)={pstop:.4f} "
                          f"({res['eot_surprisal']:.3f} nats), {pref}")
                    print(f"   stop soon: P={res['p_stop_soon']:.4f} "
                          f"(direct {res['p_stop_direct']:.4f} + via whitespace "
                          f"{''.join(res['ws_path'])!r}), then {res['after_ws_token']!r}")

            if i % 100 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)}  {el / i:.2f}s/trial  "
                      f"eta {(len(todo) - i) * el / i / 3600:.1f}h  "
                      f"deviations {n_dev} ({n_dev / i:.0%})")
    # Completion record. The invocation is written before the model loads, so an
    # interrupted run leaves an invocation claiming work it never did. Pairing
    # each invocation with a completion makes that visible: an invocation with no
    # matching completion did not finish.
    with (run_dir / "invocations.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": "completed", "n_run": len(todo), "n_non_exact": n_dev,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }) + "\n")
    print(f"[run] done. {n_dev}/{len(todo)} non-exact this invocation.")


def verify_surprisal(model, tokenizer, prompt: str, target: str,
                     wrong: str = "Purple hexagons fermented the quarterly ledger.") -> bool:
    """Sanity-check the teacher-forced surprisal indexing.

    Forced surprisal near zero for the correct target is ambiguous: it is what
    you would see both if the model is extremely confident (copying, with the
    source in context) AND if the logits are off by one so the model is scored
    on a token it has already seen. Forcing an unrelated sentence separates
    them -- a correct implementation must assign it high surprisal.
    """
    layers = [0]  # capture is irrelevant here; keep it cheap
    good = teacher_force_one(model, tokenizer, prompt, target, layers)["surprisal"]
    bad = teacher_force_one(model, tokenizer, prompt, wrong, layers)["surprisal"]
    mg, mb = sum(good) / len(good), sum(bad) / len(bad)
    print(f"  correct target: mean {mg:.4f} nats over {len(good)} tokens")
    print(f"  wrong target:   mean {mb:.4f} nats over {len(bad)} tokens")
    ok = mb > 1.0 and mb > mg * 100
    print(f"  -> indexing {'OK' if ok else 'BROKEN'}: a wrong sentence must be "
          f"surprising. {'' if ok else 'Both near zero means the model is being '
          'scored on tokens it can already see.'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="pilot", choices=["pilot", "held_out", "all"])
    ap.add_argument("--capture-layers", default="sae",
                    help="'sae' (default, 4 layers), 'all' (62), or 16,31,40,53")
    ap.add_argument("--teacher-force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--stimuli", type=Path, default=REPO_ROOT / "stimuli.csv")
    ap.add_argument("--verify-surprisal", action="store_true",
                    help="run the teacher-forced indexing check and exit")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print the frame and completion of every trial "
                         "(for small pilots; noisy above a few hundred)")
    args = ap.parse_args()

    # Everything cheap, and everything that can fail, happens BEFORE the model
    # load -- which is ~9 minutes of silence. Validating afterwards means a typo
    # in --capture-layers costs nine minutes to discover.
    t_start = time.perf_counter()
    print(f"==> 03_run_generation  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    run-id {args.run_id} | split {args.split} | "
          f"mode {'teacher-force' if args.teacher_force else 'generate'}", flush=True)

    # Layer count from the config: a JSON read, not 53 GB of weights.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    n_layers = getattr(cfg, "text_config", cfg).num_hidden_layers

    if args.capture_layers == "sae":
        layers = list(SAE_LAYERS)
    elif args.capture_layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.capture_layers.split(",")]
    if max(layers) >= n_layers:
        raise SystemExit(f"layer {max(layers)} out of range for {n_layers} layers")

    if not args.stimuli.exists():
        raise SystemExit(f"stimuli not found: {args.stimuli}")
    jobs = load_stimuli(args.stimuli, args.split)
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit(f"no stimuli for split={args.split}")

    strict, loose = load_concept_forms(REPO_ROOT / "irc" / "concepts.csv")

    run_dir = RUNS / args.run_id / ("teacher_forced" if args.teacher_force else "generated")
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_path = run_dir / "generations.jsonl"
    done = sum(1 for _ in gen_path.open()) if gen_path.exists() else 0

    # 1.09 s/trial and 6.7 MB across 62 layers, both measured (NOTES.md).
    todo = max(len(jobs) - done, 0)
    print(f"    model {MODEL_ID} ({n_layers} layers)")
    print(f"    capturing layers {layers}")
    print(f"    {args.stimuli.name}: {len(jobs):,} prompts"
          f"{f', {done:,} already done' if done else ''}")
    print(f"    -> {todo:,} to run, ~{todo * 1.09 / 3600:.2f} h, "
          f"~{todo * 6.7 * len(layers) / 62 / 1e3:.2f} GB")
    print(f"    writing to {run_dir}", flush=True)

    # Fail on VRAM here, not 60s later inside from_pretrained. A resident
    # Jupyter kernel holding the weights is the usual cause on this pod, and
    # the OOM traceback does not say so.
    import glob
    import os
    cache = os.path.join(os.environ.get("HF_HOME", ""), "hub",
                         "models--" + MODEL_ID.replace("/", "--"),
                         "snapshots", "*", "*.safetensors")
    need = sum(os.path.getsize(os.path.realpath(f)) for f in glob.glob(cache))
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"    VRAM {free / 1e9:.0f} GB free of {total / 1e9:.0f} GB"
              f"{f', weights need ~{need / 1e9:.0f} GB' if need else ''}", flush=True)
        if need and free < need * 1.05:
            raise SystemExit(
                f"\nNot enough free VRAM: {free / 1e9:.0f} GB free, "
                f"{MODEL_ID} needs ~{need / 1e9:.0f} GB.\n"
                f"Something else is holding the GPU. Run `nvidia-smi` -- a resident\n"
                f"Jupyter kernel is the usual cause; shut its kernel down, or drive\n"
                f"this module from inside that kernel instead:\n"
                f"    rg = importlib.import_module('03_run_generation')\n"
                f"    rg.run(model, tokenizer, run_dir, jobs, layers, False, *forms)")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    with (run_dir / "invocations.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": commit, "model_id": MODEL_ID,
            "split": args.split, "n_jobs": len(jobs),
            "capture_layers": layers, "teacher_force": args.teacher_force,
            "stimuli": str(args.stimuli), "limit": args.limit,
            "event": "started",
        }) + "\n")

    print("\n    loading tokenizer...", flush=True)
    tokenizer = load_tokenizer(MODEL_ID)
    print("    loading model -- ~9 min cold, ~1 min warm...", flush=True)
    t0 = time.perf_counter()
    model = load_model(MODEL_ID)
    print(f"    model loaded in {time.perf_counter() - t0:.0f}s "
          f"({time.perf_counter() - t_start:.0f}s total)\n", flush=True)

    n_actual = len(get_decoder_layers(model))
    if n_actual != n_layers:
        raise SystemExit(f"config said {n_layers} layers, model has {n_actual}")

    if args.verify_surprisal:
        print("\n=== teacher-forced surprisal indexing check ===")
        verify_surprisal(model, tokenizer, jobs[0]["prompt"], jobs[0]["target"])
        return

    if args.verbose and jobs:
        print("\n--- scaffold (constant across trials) ---")
        print(jobs[0]["prompt"])
        print("--- only the third line varies below ---")

    run(model, tokenizer, run_dir, jobs, layers, args.teacher_force, strict, loose,
        verbose=args.verbose)


if __name__ == "__main__":
    main()
