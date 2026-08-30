"""Time the generation run and preview compliance, on an already-loaded model.

Sizing the run is the point: 23,107 prompts at 2 s is 13 hours, at 5 s is 32,
and the storage decision (all 62 layers versus one) swings 200 GB. This measures
generation and activation capture separately, because capture is a second
forward pass over the full sequence and is the part worth batching if it
dominates.

It also previews the deviation rate. Upstream found ~2/3 of completions were not
exact; this fork treats that as a first-class outcome rather than an exclusion
criterion, so it is worth seeing before committing to a long run.

Capture is NOT gated on exactness. Upstream's `_generate_and_capture` does
`if exact:` before capturing, which this fork cannot do -- the pre-registered
confirmatory analysis runs on all trials.

Usage from the notebook (the model is already resident; a 27B bf16 model needs
~53 GB and the kernel is holding it, so a second copy will not fit):

    import importlib
    tc = importlib.import_module("02_timing_check")   # 02_ is not an identifier
    recs = tc.timing_check(model, tokenizer)

Standalone, only on a machine with a free GPU:

    python3 rk_scripts/02_timing_check.py --n-trials 21
"""

from irc import env  # noqa: F401  -- must be the first import

import argparse
import csv
import random
import time

import torch

from irc.model import ResidualCapture, chat_ids, get_decoder_layers
from irc.paths import REPO_ROOT


def gen_and_capture(model, tokenizer, prompt: str, target: str,
                    capture_layers: list[int]) -> dict:
    """One trial: greedy generation, then all-layer capture on response tokens.

    Mirrors the runner exactly, including capturing for non-exact completions.
    Returns the completion, its exactness, and the two timings separately.
    """
    ids = chat_ids(tokenizer, prompt)
    n_prompt = ids.shape[1]
    tgt_len = len(tokenizer(target, add_special_tokens=False)["input_ids"])

    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=tgt_len + 16, do_sample=False)
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0

    gen = out[0, n_prompt:]
    end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ends = (gen == end_id).nonzero()
    n_resp = int(ends[0]) if len(ends) else len(gen)
    completion = tokenizer.decode(gen[:n_resp], skip_special_tokens=True).strip()

    t0 = time.perf_counter()
    with ResidualCapture(model, capture_layers) as cap:
        model(out[:, : n_prompt + n_resp])
    acts = torch.stack([cap.acts[i][0, n_prompt : n_prompt + n_resp]
                        for i in capture_layers]).to(torch.bfloat16)
    torch.cuda.synchronize()
    t_cap = time.perf_counter() - t0

    return {"completion": completion, "exact": completion == target,
            "n_resp": n_resp, "t_gen": t_gen, "t_cap": t_cap,
            "mb": acts.nbytes / 1e6}


def timing_check(model, tokenizer, stimuli_path=None, n_trials: int = 21,
                 seed: int = 0, verbose: bool = True) -> list[dict]:
    """Run n_trials sampled across all phrasings and print a sizing report.

    The first trial is discarded as CUDA warmup. Sampling is across the whole
    stimulus file rather than one condition, so prompt lengths span the real
    18-40 token range instead of clustering.
    """
    stimuli_path = stimuli_path or REPO_ROOT / "stimuli.csv"
    layers = list(range(len(get_decoder_layers(model))))
    with open(stimuli_path) as fh:
        rows = list(csv.DictReader(fh))
    sample = random.Random(seed).sample(rows, n_trials)

    recs = []
    for i, row in enumerate(sample):
        rec = gen_and_capture(model, tokenizer, row["prompt"], row["target"], layers)
        rec["phrasing"] = row["phrasing_id"]
        rec["cell"] = row["cell_id"]
        recs.append(rec)
        if verbose and i == 0:
            print(f"(warmup discarded: {rec['t_gen'] + rec['t_cap']:.2f}s)")

    if verbose:
        report(recs[1:], len(layers))
    return recs


def report(recs: list[dict], n_layers: int) -> None:
    n = len(recs)
    t_gen = sum(r["t_gen"] for r in recs) / n
    t_cap = sum(r["t_cap"] for r in recs) / n
    mb = sum(r["mb"] for r in recs) / n
    per = t_gen + t_cap

    print(f"\nper trial: {per:.2f}s  (generate {t_gen:.2f} + capture {t_cap:.2f}, "
          f"capture is {t_cap / per:.0%})")
    print(f"{mb:.1f} MB/trial across {n_layers} layers; "
          f"response tokens {min(r['n_resp'] for r in recs)}-"
          f"{max(r['n_resp'] for r in recs)}")
    print(f"exact completions: {sum(r['exact'] for r in recs)}/{n}")

    print(f"\n{'':<18}{'hours':>8}{'GB all':>9}{'GB 1-layer':>12}")
    for label, count in (("pilot 4,690", 4690), ("held out 18,760", 18760),
                         ("full 23,107", 23107)):
        print(f"  {label:<16}{count * per / 3600:8.1f}{count * mb / 1e3:9.1f}"
              f"{count * mb / n_layers / 1e3:12.2f}")

    bad = [r for r in recs if not r["exact"]]
    if bad:
        print(f"\nnon-exact completions ({len(bad)}/{n}):")
        for r in bad:
            print(f"  {r['phrasing']:4s} {r['completion'][:88]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    if free / 1e9 < 55:
        raise SystemExit(
            f"only {free / 1e9:.0f} GB free of {total / 1e9:.0f} GB -- a 27B bf16 "
            f"model needs ~53 GB. Something else is holding the GPU (check "
            f"nvidia-smi); import this module into that process instead.")

    from irc.model import MODEL_ID, load_model, load_tokenizer
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID)
    timing_check(model, tokenizer, n_trials=args.n_trials, seed=args.seed)


if __name__ == "__main__":
    main()
