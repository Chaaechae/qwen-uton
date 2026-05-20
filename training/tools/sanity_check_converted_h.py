"""
Sanity-check: confirm the converted H checkpoint loads into the same
PointTransformerV3 architecture as the published utonia.pth.

Both files, after `utonia.load(...)`, should produce models whose
state_dict keys and parameter shapes are IDENTICAL.  Weight values
should DIFFER (H was fine-tuned 5+ epochs from the utonia.pth
warm-start, so the backbone moved a bit), but never by an absurd
amount — typical per-param L1 diff on the order of 1e-2 to 1e-1
indicates "same architecture, slightly different weights".

A diff of zero everywhere would mean we accidentally loaded the
same checkpoint twice (or the H training never updated the
backbone).  A diff of NaN / inf or > 10 mean something is broken
in the conversion script.

Usage:
    python tools/sanity_check_converted_h.py \
        --h-ckpt    exp/utonia_q35_h/inference_ckpt.pth \
        --ref-ckpt  /group-volume/Utonia/utonia.pth
"""

import argparse
import sys
import torch
import utonia


def load_model(path):
    print(f"[load] {path}")
    return utonia.load(path)


def compare(model_h, model_u):
    sd_h = model_h.state_dict()
    sd_u = model_u.state_dict()
    keys_h = set(sd_h.keys())
    keys_u = set(sd_u.keys())

    only_in_h = sorted(keys_h - keys_u)
    only_in_u = sorted(keys_u - keys_h)
    common = sorted(keys_h & keys_u)

    print(f"\n[keys]  H={len(keys_h)}  utonia={len(keys_u)}  "
          f"common={len(common)}")
    if only_in_h:
        print(f"  only in H ({len(only_in_h)}):")
        for k in only_in_h[:5]:
            print(f"    {k}")
        if len(only_in_h) > 5:
            print(f"    ... +{len(only_in_h)-5} more")
    if only_in_u:
        print(f"  only in utonia ({len(only_in_u)}):")
        for k in only_in_u[:5]:
            print(f"    {k}")
        if len(only_in_u) > 5:
            print(f"    ... +{len(only_in_u)-5} more")

    # Shape comparison on shared keys.
    shape_mismatch = []
    for k in common:
        if sd_h[k].shape != sd_u[k].shape:
            shape_mismatch.append((k, tuple(sd_h[k].shape), tuple(sd_u[k].shape)))
    print(f"\n[shapes] mismatched: {len(shape_mismatch)}")
    for k, s_h, s_u in shape_mismatch[:5]:
        print(f"  {k}:  H={s_h}  utonia={s_u}")

    # Value-diff stats on shared keys with matching shapes.
    print(f"\n[values] L1 diff per parameter group (top 10 largest):")
    diffs = []
    for k in common:
        if sd_h[k].shape != sd_u[k].shape:
            continue
        a = sd_h[k].float()
        b = sd_u[k].float()
        if a.numel() == 0:
            continue
        l1 = float((a - b).abs().mean())
        if not (l1 == l1):  # NaN
            print(f"  WARN: NaN diff at {k}")
            continue
        diffs.append((k, l1, float(a.abs().mean()), float(b.abs().mean())))

    diffs.sort(key=lambda x: -x[1])
    for k, l1, m_h, m_u in diffs[:10]:
        print(f"  {l1:.4e}   {k:55s}   |H|={m_h:.3f}  |U|={m_u:.3f}")
    if not diffs:
        print("  (no comparable parameter groups)")
        return

    overall = sum(d[1] for d in diffs) / len(diffs)
    zero_count = sum(1 for d in diffs if d[1] == 0.0)
    print(f"\n  mean L1 over {len(diffs)} groups = {overall:.4e}")
    print(f"  groups with exactly-zero diff = {zero_count}/{len(diffs)}")

    # Quick health check.
    print("\n[verdict]")
    if only_in_h or only_in_u:
        print("  ✗ architecture mismatch — keys do not match")
        sys.exit(1)
    if shape_mismatch:
        print("  ✗ architecture mismatch — shape mismatch on shared keys")
        sys.exit(1)
    if overall == 0:
        print("  ⚠ identical weights — same file loaded twice, or training "
              "did NOT update the backbone")
        sys.exit(1)
    if overall > 5:
        print(f"  ⚠ very large mean diff ({overall:.2f}) — investigate")
    print("  ✓ same architecture; weights differ as expected from training")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h-ckpt",   required=True,
                   help="Path to converted H checkpoint "
                        "(output of convert_h_ckpt_for_utonia_inference.py).")
    p.add_argument("--ref-ckpt", default="/group-volume/Utonia/utonia.pth",
                   help="Path to reference utonia.pth.")
    args = p.parse_args()

    model_h = load_model(args.h_ckpt)
    model_u = load_model(args.ref_ckpt)

    # Quick forward shape check — both should accept the same input format.
    print("\n[forward] dummy forward shape check ...")
    import numpy as np
    coord = torch.from_numpy(np.random.randn(1000, 3).astype(np.float32))
    color = torch.from_numpy(np.random.rand(1000, 3).astype(np.float32))
    normal = torch.from_numpy(np.random.randn(1000, 3).astype(np.float32))
    grid_coord = (coord / 0.02).long()
    feat = torch.cat([coord, color, normal], dim=-1)
    offset = torch.tensor([1000], dtype=torch.long)
    batch = torch.zeros(1000, dtype=torch.long)
    point_dict = dict(
        coord=coord, color=color, normal=normal, grid_coord=grid_coord,
        feat=feat, offset=offset, batch=batch,
    )

    for name, m in [("H", model_h), ("utonia", model_u)]:
        try:
            point = utonia.structure.Point(point_dict)
            out = m(point)
            while "pooling_parent" in out.keys():
                parent = out.pop("pooling_parent")
                inv = out.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, out.feat[inv]], dim=-1)
                out = parent
            print(f"  [{name}] forward OK   "
                  f"out.feat.shape={tuple(out.feat.shape)}")
        except Exception as e:
            print(f"  [{name}] FORWARD ERROR: {type(e).__name__}: {e}")
            sys.exit(1)

    compare(model_h, model_u)


if __name__ == "__main__":
    main()
