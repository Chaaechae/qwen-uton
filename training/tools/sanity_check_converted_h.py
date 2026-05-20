"""
Sanity-check: confirm the converted H checkpoint is structurally
identical to the published utonia.pth.

What "structurally identical" means
-----------------------------------
Both files should hold a `state_dict` whose KEYS and SHAPES are an
exact match.  Weight VALUES will differ because H was fine-tuned
from the utonia.pth warm-start (5+ alignment epochs nudged the
backbone toward the Qwen direction).  Typical per-parameter L1
diff sits between 1e-3 and 1e-1.

This is a *file-level* comparison — it does NOT instantiate the
PointTransformerV3 model, so the script has no dependency on the
`utonia` package being importable.  All we need is `torch`.

Usage:
    python tools/sanity_check_converted_h.py \
        --h-ckpt    exp/utonia_q35_h/inference_ckpt.pth \
        --ref-ckpt  /group-volume/Utonia/utonia.pth
"""

import argparse
import sys
import torch


def load_state_dict(path):
    print(f"[load] {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        cfg = ckpt.get("config", None)
    else:
        sd = ckpt
        cfg = None
    # Normalise potential `module.` DDP prefix so both files compare in
    # the same namespace.  Our converter writes raw PT-v3 keys (no
    # `module.`); utonia.pth ships with `module.` on every key.
    norm = {}
    for k, v in sd.items():
        if k.startswith("module."):
            norm[k[len("module."):]] = v
        else:
            norm[k] = v
    return norm, cfg


def compare_configs(cfg_h, cfg_u):
    if cfg_h is None and cfg_u is None:
        print("[config] neither file has a 'config' dict — skipping")
        return
    if cfg_h is None or cfg_u is None:
        which = "H" if cfg_h is None else "utonia"
        print(f"[config] only one file has 'config' (the other is {which}) "
              "— skipping comparison")
        return
    print("\n[config] both files have a 'config' dict; comparing kwargs ...")
    keys = sorted(set(cfg_h.keys()) | set(cfg_u.keys()))
    mismatches = 0
    for k in keys:
        v_h = cfg_h.get(k, "<missing>")
        v_u = cfg_u.get(k, "<missing>")
        if v_h != v_u:
            print(f"  diff: {k}:  H={v_h!r}  utonia={v_u!r}")
            mismatches += 1
    if mismatches == 0:
        print(f"  all {len(keys)} config kwargs match")


def compare_state_dicts(sd_h, sd_u):
    keys_h = set(sd_h.keys())
    keys_u = set(sd_u.keys())
    only_in_h = sorted(keys_h - keys_u)
    only_in_u = sorted(keys_u - keys_h)
    common = sorted(keys_h & keys_u)

    print(f"\n[keys]  H={len(keys_h)}  utonia={len(keys_u)}  "
          f"common={len(common)}")
    if only_in_h:
        print(f"  only in H ({len(only_in_h)}):")
        for k in only_in_h[:8]:
            print(f"    {k}")
        if len(only_in_h) > 8:
            print(f"    ... +{len(only_in_h) - 8} more")
    if only_in_u:
        print(f"  only in utonia ({len(only_in_u)}):")
        for k in only_in_u[:8]:
            print(f"    {k}")
        if len(only_in_u) > 8:
            print(f"    ... +{len(only_in_u) - 8} more")

    # Shape check on shared keys.
    shape_mismatch = []
    for k in common:
        if sd_h[k].shape != sd_u[k].shape:
            shape_mismatch.append((k, tuple(sd_h[k].shape), tuple(sd_u[k].shape)))
    print(f"\n[shapes] mismatched among shared keys: {len(shape_mismatch)}")
    for k, s_h, s_u in shape_mismatch[:5]:
        print(f"  {k}:  H={s_h}  utonia={s_u}")

    # Value-diff stats (only over shared keys with matching shapes).
    diffs = []
    for k in common:
        if sd_h[k].shape != sd_u[k].shape:
            continue
        a = sd_h[k].float()
        b = sd_u[k].float()
        if a.numel() == 0:
            continue
        l1 = float((a - b).abs().mean())
        if not (l1 == l1):
            print(f"  WARN: NaN diff at {k}")
            continue
        diffs.append((k, l1, float(a.abs().mean()), float(b.abs().mean())))

    diffs.sort(key=lambda x: -x[1])
    print(f"\n[values] L1 diff per parameter group (top 10 largest):")
    for k, l1, m_h, m_u in diffs[:10]:
        print(f"  {l1:.4e}   {k:55s}   |H|={m_h:.3f}  |U|={m_u:.3f}")
    if not diffs:
        print("  (no comparable parameter groups)")
        return only_in_h, only_in_u, shape_mismatch, 0.0, 0

    overall = sum(d[1] for d in diffs) / len(diffs)
    zero_count = sum(1 for d in diffs if d[1] == 0.0)
    print(f"\n  mean L1 over {len(diffs)} groups = {overall:.4e}")
    print(f"  exactly-zero groups = {zero_count}/{len(diffs)}")
    return only_in_h, only_in_u, shape_mismatch, overall, zero_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h-ckpt",   required=True,
                   help="Converted H checkpoint "
                        "(output of convert_h_ckpt_for_utonia_inference.py).")
    p.add_argument("--ref-ckpt", default="/group-volume/Utonia/utonia.pth")
    args = p.parse_args()

    sd_h, cfg_h = load_state_dict(args.h_ckpt)
    sd_u, cfg_u = load_state_dict(args.ref_ckpt)

    compare_configs(cfg_h, cfg_u)
    only_in_h, only_in_u, shape_mismatch, mean_diff, zero_count = (
        compare_state_dicts(sd_h, sd_u)
    )

    print("\n[verdict]")
    if only_in_h or only_in_u:
        print("  ✗ key mismatch — architectures differ")
        sys.exit(1)
    if shape_mismatch:
        print("  ✗ shape mismatch — architectures differ")
        sys.exit(1)
    if mean_diff == 0:
        print("  ⚠ identical weights — same file twice, or training never "
              "updated the backbone")
        sys.exit(1)
    if mean_diff > 5:
        print(f"  ⚠ very large mean diff ({mean_diff:.2f}) — investigate")
    print(f"  ✓ same architecture; weights differ as expected from "
          f"alignment fine-tune (mean L1 diff {mean_diff:.4e})")


if __name__ == "__main__":
    main()
