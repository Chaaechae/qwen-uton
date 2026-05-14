from .utonia_v1m1_base import *
from .utonia_v1m2_qwen3_5_distill import *
from .utonia_v1m3a_qwen3_5_align_only import *
from .utonia_v1m3b_qwen3_5_distill_ema import *

# Side-effect import: registers SkipOnErrorImagePointDataset in the DATASETS
# registry. The file physically lives in pointcept/datasets/ but we trigger
# its import from here so Pointcept's pointcept/datasets/__init__.py doesn't
# need to be modified.
from pointcept.datasets.skip_on_error_dataset import *  # noqa: F401, E402

# Monkey-patch Config.dump to be non-fatal on yapf failure.
# Pointcept's `default_config_parser` calls `cfg.dump(<save_path>/config.py)`
# at startup. That dump generates the cfg as a Python source string and runs
# yapf on it. yapf occasionally crashes (YapfError / SyntaxError) on certain
# generated tokens, killing the run *before training starts* — even though
# the dump is purely for reproducibility (the trainer reads cfg directly).
#
# This wrapper catches yapf failures, prints a warning, and falls back to a
# plain `repr` of the cfg dict so the run can continue.
import pointcept.utils.config as _ptconfig  # noqa: E402

if not getattr(_ptconfig.Config.dump, "_yapf_safe", False):
    _orig_dump = _ptconfig.Config.dump

    def _safe_dump(self, file=None):
        try:
            return _orig_dump(self, file)
        except Exception as e:
            print(
                f"[Config.dump] yapf failed ({type(e).__name__}: "
                f"{str(e)[:200]}). Falling back to plain repr.",
                flush=True,
            )
            cfg_dict = (
                super(_ptconfig.Config, self)
                .__getattribute__("_cfg_dict")
                .to_dict()
            )
            text = repr(cfg_dict)
            if file is None:
                return text
            with open(file, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            return None

    _safe_dump._yapf_safe = True
    _ptconfig.Config.dump = _safe_dump
