"""
SkipOnErrorImagePointDataset
============================

Drop-in replacement for `DefaultImagePointDataset` that does NOT crash
on per-sample exceptions. When `__getitem__` raises, the failure is
logged (one JSONL line) and the iterator advances to the next index.
After `max_retries` consecutive failures, we give up and raise — that
state probably means a real bug, not a single bad sample.

Why this exists
---------------
PyTorch DataLoader has no skip-on-error semantics; Pointcept's
`DefaultImagePointDataset.__getitem__` has no `try/except` wrapper.
One bad scene (missing file / truncated PNG / empty crop / ...)
brings down a multi-hour run with a `_next_data` traceback.

Usage
-----
In the training config, swap the dataset `type`:

    dict(
        type="SkipOnErrorImagePointDataset",
        crop_h=crop_h, crop_w=crop_w, patch_size=patch_size,
        split=["train", "val"],
        data_root=f"{DATASET_ROOT}/data/scannet",
        transform=indoor_transform,
        test_mode=False,
        loop=1,
        # Optional:
        # max_retries=10,
        # skip_log_path="exp/.../skip_log.jsonl",
    ),

All other arguments are forwarded to the parent class unchanged.

Log location
------------
Resolution order for the path:
    1. `skip_log_path` argument in the config
    2. `UTONIA_SKIP_LOG_PATH` env var
    3. `./skip_log.jsonl` (= `third_party/Pointcept/skip_log.jsonl`)

The rank is always appended to the filename (`<base>_rank{N}.jsonl`)
so DataLoader workers across ranks don't fight for the same file.

Log format (one JSON object per line)
-------------------------------------
    {
      "iso_time":      "2026-05-14T08:23:11",
      "idx":           127,
      "name":          "scene0023_00_3",
      "split":         "train",
      "error_type":    "FileNotFoundError",
      "error_message": "..."
    }
"""

import os
import json
import datetime

import torch.distributed as dist

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultImagePointDataset


__all__ = ["SkipOnErrorImagePointDataset"]


@DATASETS.register_module("SkipOnErrorImagePointDataset")
class SkipOnErrorImagePointDataset(DefaultImagePointDataset):
    def __init__(self, *args, max_retries=10, skip_log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries

        rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        if skip_log_path is None:
            skip_log_path = os.environ.get(
                "UTONIA_SKIP_LOG_PATH", "skip_log.jsonl"
            )
        base, ext = os.path.splitext(skip_log_path)
        ext = ext or ".jsonl"
        self.skip_log_path = f"{base}_rank{rank}{ext}"

        log_dir = os.path.dirname(self.skip_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------ utils
    def _log_skip(self, idx, exc):
        try:
            name = self.get_data_name(idx)
        except Exception:
            name = None
        try:
            split = self.get_split_name(idx)
        except Exception:
            split = None
        record = {
            "iso_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "idx": int(idx),
            "name": name,
            "split": split,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:400],
        }
        try:
            # Single-line append; atomic on local FS for sub-PIPE_BUF writes,
            # even from multiple DataLoader workers.
            with open(self.skip_log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Don't let the logger itself crash training.
            pass

    # --------------------------------------------------------------- __getitem__
    def __getitem__(self, idx):
        # Test pipeline (test_voxelize / test_crop) is more complex; any error
        # there usually indicates a real bug — let it surface normally.
        if self.test_mode:
            return self.prepare_test_data(idx)

        original_idx = idx
        attempts_left = self.max_retries
        seen = set()
        while attempts_left > 0:
            if idx in seen:
                idx = (idx + 1) % len(self)
                continue
            seen.add(idx)
            try:
                return self.prepare_train_data(idx)
            except Exception as e:
                self._log_skip(idx, e)
                try:
                    name = self.get_data_name(idx)
                except Exception:
                    name = "?"
                print(
                    f"[SkipOnError] sample {idx} ({name}) failed: "
                    f"{type(e).__name__}: {str(e)[:160]}",
                    flush=True,
                )
                attempts_left -= 1
                idx = (idx + 1) % len(self)

        raise RuntimeError(
            f"More than {self.max_retries} consecutive bad samples starting "
            f"from idx {original_idx}. See {self.skip_log_path} for details."
        )
