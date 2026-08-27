import os, torch
from transformers import AutoConfig
from sae_lens import SAE
import config
from huggingface_hub import snapshot_download, scan_cache_dir


# 1. cache is on the volume, not the container disk
assert os.environ.get("HF_HOME", "").startswith("/workspace"), \
    f"HF_HOME={os.environ.get('HF_HOME')!r} — weights would die with the pod"

# 2. GPU is present and big enough
assert torch.cuda.is_available(), "no CUDA"
gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"gpu: {torch.cuda.get_device_name(0)}  {gib:.0f} GiB")
assert gib > 70, "need an 80GB card for the 27B in bf16"

# 3. every profile: model config, layer depth, and SAEs all agree
for mode, p in config.PROFILES.items():
    mcfg = AutoConfig.from_pretrained(p["model_id"])
    d_model  = getattr(mcfg, "hidden_size", None)  or mcfg.text_config.hidden_size
    n_layers = getattr(mcfg, "num_hidden_layers", None) or mcfg.text_config.num_hidden_layers
    depth = p["layer"] / n_layers
    print(f"\n[{mode}] {p['model_id']}  d_model={d_model} layers={n_layers} "
          f"layer={p['layer']} ({depth:.0%} depth)")
    assert p["layer"] < n_layers

    for name, sid in p["saes"].items():
        sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(release=p["sae_release"], sae_id=sid)
        assert sae.cfg.d_in == d_model, f"{sid}: d_in {sae.cfg.d_in} != {d_model}"
        print(f"  ok {name:6s} {sid}  d_sae={sae.cfg.d_sae}")
        del sae
        torch.cuda.empty_cache()

print("\nall profiles OK")

# 4. make sure all the models I expect to have cached are

def is_cached(repo_id):
    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        return False

print("\nweights on disk:")
wanted = [p["model_id"] for p in config.PROFILES.values()] + config.EXTRA_MODELS
missing = []
for repo in wanted:
    ok = is_cached(repo)
    print(f"  {'ok  ' if ok else 'MISS'} {repo}")
    if not ok:
        missing.append(repo)

cache_gb = scan_cache_dir().size_on_disk / 1024**3
print(f"\nHF cache: {cache_gb:.1f} GB of 200 GB")
if cache_gb > 150:
    print("  ⚠ past the 150 GB tripwire — expand the volume before the next big download")

assert not missing, f"not downloaded: {missing}"