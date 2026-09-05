"""Pack a run's per-trial activation files into one tensor plus an index.

Why: a pass stores one small `.pt` per trial -- 27,674 files across the four
passes, ~301 KB each. That shape is bad in three separate ways.

  storage   ~8.3 GB of tensor data occupies ~15 GB on the network filesystem,
            because each tiny file rounds up to a block.
  reading   MooseFS costs most of a second per file open, so a sequential
            reader gets ~1 file/s; the measure stage needed a 32-thread
            prefetch to reach ~12 trials/s (see NOTES.md 2026-08-31).
  archiving anything that moves these bytes -- git, LFS, HF Hub, object
            storage -- handles 4 large files far better than 27,674 small ones,
            and git in particular walks the whole tree on every operation.

Trials have different token counts, so the pack CONCATENATES along the token
axis rather than padding: one (n_layers, total_tokens, d_model) tensor plus an
index giving each trial's offset and length. Padding would have to invent a mask
and would waste space on the longest trial.

Round-trips exactly: `--verify` reloads every trial from the pack and asserts
bitwise equality against the original file.

    python3 rk_scripts/16_consolidate_acts.py --run-id pilot2 --pass generated
    python3 rk_scripts/16_consolidate_acts.py --run-id heldout1 --all-passes --verify

Writes `acts_packed.pt` and `acts_index.json` beside `generations.jsonl`. The
originals are left alone -- deleting them is a separate, deliberate step, and
`--verify` should pass first.
"""

from irc import env  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import torch

from irc.paths import RUNS


def pack(run_dir: Path, workers: int = 32) -> tuple[Path, Path]:
    """Concatenate every trial's activations into one tensor plus an index."""
    from concurrent.futures import ThreadPoolExecutor

    recs = []
    with (run_dir / "generations.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("acts_file"):
                recs.append((r["prompt_group"], r["acts_file"]))
    # last occurrence wins, matching 09_measure.py's load_records
    seen = {}
    for pg, f in recs:
        seen[pg] = f
    items = sorted(seen.items())
    print(f"{run_dir}: {len(items):,} trials to pack")

    def load(item):
        pg, f = item
        return pg, torch.load(run_dir / f, map_location="cpu")

    tensors, index, offset = [], [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        window = workers * 8
        for start in range(0, len(items), window):
            for pg, A in pool.map(load, items[start:start + window]):
                if A.ndim != 3:
                    raise SystemExit(f"{pg}: expected (layers, tokens, d), "
                                     f"got {tuple(A.shape)}")
                n_tok = A.shape[1]
                tensors.append(A)
                index.append({"prompt_group": pg, "start": offset,
                              "length": n_tok})
                offset += n_tok
            done = min(start + window, len(items))
            if done % 4096 < window:
                el = time.time() - t0
                print(f"  {done:,}/{len(items):,}  {done / max(el, 1e-9):.0f} "
                      f"trials/s", flush=True)

    if not tensors:
        raise SystemExit("no activations found")
    n_layers, _, d_model = tensors[0].shape
    if any(t.shape[0] != n_layers or t.shape[2] != d_model for t in tensors):
        raise SystemExit("inconsistent layer count or d_model across trials")

    packed = torch.cat(tensors, dim=1)
    print(f"  packed tensor {tuple(packed.shape)} {packed.dtype} "
          f"= {packed.numel() * packed.element_size() / 1e9:.2f} GB")

    out_pt = run_dir / "acts_packed.pt"
    out_ix = run_dir / "acts_index.json"
    torch.save(packed, out_pt)
    out_ix.write_text(json.dumps(
        {"n_layers": n_layers, "d_model": d_model,
         "dtype": str(packed.dtype), "total_tokens": int(packed.shape[1]),
         "trials": index}, indent=1))
    print(f"  wrote {out_pt.name} and {out_ix.name}")
    return out_pt, out_ix


def verify(run_dir: Path, limit: int = 0) -> bool:
    """Reload each trial from the pack and assert bitwise equality."""
    ix = json.loads((run_dir / "acts_index.json").read_text())
    packed = torch.load(run_dir / "acts_packed.pt", map_location="cpu")
    files = {}
    with (run_dir / "generations.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("acts_file"):
                files[r["prompt_group"]] = r["acts_file"]

    # Parallel, for the same reason the measure stage prefetches: a sequential
    # reader gets ~1 file/s here, which would make a full verify take hours.
    from concurrent.futures import ThreadPoolExecutor

    todo = [t for t in ix["trials"] if t["prompt_group"] in files]
    if limit:
        todo = todo[:limit]

    def check(t):
        pg = t["prompt_group"]
        orig = torch.load(run_dir / files[pg], map_location="cpu")
        got = packed[:, t["start"]:t["start"] + t["length"], :]
        ok = got.shape == orig.shape and torch.equal(got, orig)
        return pg, ok, tuple(got.shape), tuple(orig.shape)

    bad, checked = 0, 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for pg, ok, gs, os_ in pool.map(check, todo):
            if not ok:
                bad += 1
                if bad <= 3:
                    print(f"  MISMATCH {pg}: {gs} vs {os_}")
            checked += 1
            if checked % 4000 == 0:
                print(f"  verified {checked:,}/{len(todo):,}...", flush=True)
    print(f"  verified {checked:,} trials, {bad} mismatches")
    return bad == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pass", dest="pass_", default="generated")
    ap.add_argument("--all-passes", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="reload every trial from the pack and compare bitwise")
    ap.add_argument("--verify-limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    passes = (["generated", "teacher_forced"] if args.all_passes
              else [args.pass_])
    ok = True
    for p in passes:
        run_dir = RUNS / args.run_id / p
        if not (run_dir / "generations.jsonl").exists():
            print(f"skip {args.run_id}/{p}: no generations.jsonl")
            continue
        print(f"\n=== {args.run_id}/{p} ===")
        pack(run_dir, args.workers)
        if args.verify:
            ok &= verify(run_dir, args.verify_limit)
    if args.verify:
        print(f"\nverification {'PASSED' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit(1)
        print("Originals are untouched. Deleting them is a separate, "
              "deliberate step.")


if __name__ == "__main__":
    main()
