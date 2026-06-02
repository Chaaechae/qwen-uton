"""
Attention video: fixed scene point cloud, sweep many 2D frames, animate how the 3D
heatmap (which scene region each image attends to) changes frame to frame.

Output: a self-contained plotly HTML with Play + a frame slider (left = the input
image, right = the 3D scene colored by per-point match to that image). Optionally an
.mp4/.gif if kaleido + imageio are available (--mp4 / --gif).

Modes:
    image (default) : per-point max-cosine over ALL DINO patches -> the image footprint
    text --text "X" : CLIPSeg region for "X" per frame -> where that object is in 3D

Run (no conda activate; repo root auto-added to sys.path):
    python tools/viewpoint_probe/attention_video.py \
        --scene_dir /group-volume/3Ddataset/data/scannet/val/scene0011_00 \
        --max_frames 40 --out /tmp/scene0011_attn
"""

import argparse
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F

import probe_2d3d_alignment as P
import localize_from_2d as L


def _norm01(x):
    return (x - x.min()) / (x.ptp() + 1e-6)


def _small(img, maxw=384):
    step = max(1, int(np.ceil(img.shape[1] / maxw)))
    return np.ascontiguousarray(img[::step, ::step])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone_ckpt",
                    default="/group-volume/Utonia/pretrain-utonia-v1m1-0-base-stagev2.pth")
    ap.add_argument("--patch_proj_ckpt", default=None)
    ap.add_argument("--scene_dir",
                    default="/group-volume/3Ddataset/data/scannet/val/scene0011_00")
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--frames", default="", help="comma frame ids; empty = all in scene")
    ap.add_argument("--max_frames", type=int, default=40, help="even-subsample to this many")
    ap.add_argument("--mode", choices=["image", "text"], default="image")
    ap.add_argument("--text", default=None, help="prompt for --mode text")
    ap.add_argument("--text_topp", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max_points", type=int, default=40000,
                    help="point subsample for a responsive/small HTML")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--mp4", action="store_true", help="also render an .mp4 (needs kaleido+imageio)")
    ap.add_argument("--gif", action="store_true", help="also render a .gif (needs kaleido+imageio)")
    ap.add_argument("--out", default="/tmp/scene_attn")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = P.load_backbone(args.backbone_ckpt, device)
    patch_proj = P.load_patch_proj(args.patch_proj_ckpt or args.backbone_ckpt, device)
    dino = P.load_dino(device)
    F3D, scene = P.compute_scene_F3D(args.scene_dir, model, patch_proj, device, args.scale)
    coord = scene["coord"]
    N = len(coord)
    image_dir = args.image_dir or P.default_image_dir(args.scene_dir)
    print(f"[scene] N={N}  F3D={tuple(F3D.shape)}  image_dir={image_dir}")

    frames = args.frames.split(",") if args.frames else P.list_frames(image_dir)
    if args.max_frames > 0 and len(frames) > args.max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, args.max_frames).astype(int)]
    print(f"[frames] {len(frames)}")

    # fixed point subsample shared by every frame (geometry constant)
    idx = L._subsample(N, args.max_points)
    co = coord[idx]

    import imageio.v2 as imageio
    imgs, scs = [], []
    for fid in frames:
        png = os.path.join(image_dir, "color", f"{fid}.png")
        if not os.path.isfile(png):
            continue
        rgb = imageio.imread(png)
        img_t, sx, sy = P.dino_preprocess(rgb)
        F2D = F.normalize(P.extract_dino_patches(dino, img_t, device), dim=-1)
        if args.mode == "text":
            assert args.text, "--text required for --mode text"
            kp, heat37 = L.clip_text_region(rgb, args.text, device, args.text_topp)
            Fk = F2D[torch.as_tensor(kp, device=device)]
            scores = P._max_sim_scores(F3D, Fk)
            rgb = L.highlight_image(rgb, heat37)
        else:
            scores = P._max_sim_scores(F3D, F2D)            # whole-image footprint
        imgs.append(_small(rgb))
        scs.append(_norm01(scores)[idx].astype(np.float32))
        print(f"  frame {fid}: scored")
    assert scs, "no frames scored"

    _write_html(args.out + ".html", co, imgs, scs, frames, args)
    print(f"[out] {args.out}.html  (open in a browser; Play / slider)")
    if args.mp4 or args.gif:
        _write_video(args.out, co, imgs, scs, frames, args)


def _write_html(path, co, imgs, scs, frames, args):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.36, 0.64],
                        specs=[[{"type": "image"}, {"type": "scene"}]],
                        subplot_titles=("input frame", "3D attention (bright = match)"))
    fig.add_trace(go.Image(z=imgs[0]), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=co[:, 0], y=co[:, 1], z=co[:, 2], mode="markers",
        marker=dict(size=1.6, color=scs[0], colorscale="Spectral_r",
                    cmin=0.0, cmax=1.0, opacity=0.85)), row=1, col=2)
    # partial frame updates (only image z + marker color) -> small file.
    # NOTE: dicts MUST carry "type" or plotly validates them as a 2D Scatter and
    # rejects 'z' ("invalid property ... Scatter: 'z'").
    fig.frames = [go.Frame(name=str(f), data=[
                      dict(type="image", z=imgs[i]),
                      dict(type="scatter3d",
                           marker=dict(color=scs[i], colorscale="Spectral_r",
                                       cmin=0, cmax=1, size=1.6, opacity=0.85))],
                           traces=[0, 1]) for i, f in enumerate(frames[:len(scs)])]
    dur = int(1000 / max(args.fps, 1))
    fig.update_layout(
        title=f"{args.mode}" + (f" '{args.text}'" if args.mode == "text" else "")
              + f" | {len(scs)} frames",
        scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0,
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=dur, redraw=True),
                                                    fromcurrent=True)]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.05, len=0.9, steps=[
            dict(label=str(f), method="animate",
                 args=[[str(f)], dict(mode="immediate",
                       frame=dict(duration=0, redraw=True))])
            for f in frames[:len(scs)]])])
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def _write_video(out, co, imgs, scs, frames, args):
    """Optional raster video via kaleido (static 3D render) + imageio."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import imageio.v2 as imageio
    except Exception as e:
        print(f"[video] skipped ({e})")
        return
    cam = dict(eye=dict(x=1.5, y=1.5, z=1.2))   # fixed camera so the scene doesn't spin
    pngs = []
    for i in range(len(scs)):
        fig = make_subplots(rows=1, cols=2, column_widths=[0.36, 0.64],
                            specs=[[{"type": "image"}, {"type": "scene"}]])
        fig.add_trace(go.Image(z=imgs[i]), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=co[:, 0], y=co[:, 1], z=co[:, 2], mode="markers",
            marker=dict(size=1.6, color=scs[i], colorscale="Spectral_r",
                        cmin=0, cmax=1, opacity=0.85)), row=1, col=2)
        fig.update_layout(scene=dict(aspectmode="data", camera=cam),
                          margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        try:
            pngs.append(imageio.imread(fig.to_image(format="png", width=1100, height=620,
                                                    engine="kaleido")))
        except Exception as e:
            print(f"[video] kaleido render failed ({e}); install kaleido. Skipping mp4/gif.")
            return
    if args.mp4:
        imageio.mimsave(out + ".mp4", pngs, fps=args.fps)
        print(f"[out] {out}.mp4")
    if args.gif:
        imageio.mimsave(out + ".gif", pngs, fps=args.fps)
        print(f"[out] {out}.gif")


if __name__ == "__main__":
    main()
