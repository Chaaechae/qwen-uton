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
# yapf on it. yapf occasionally crashes (YapfError / SyntaxError / lib2to3
# ParseError) on certain generated tokens, killing the run *before training
# starts* — even though the dump is purely for reproducibility (the trainer
# reads cfg directly).
#
# We apply TWO layers of defense:
#   1. yapf.yapflib.yapf_api.FormatCode  → wrapped so failure returns text
#      unchanged instead of raising. This actually fixes the immediate cause.
#   2. pointcept.utils.config.Config.dump → wrapped so any other failure path
#      (including bugs in `_format_dict`) falls back to a plain `repr` of the
#      cfg dict and the run continues.
import pointcept.utils.config as _ptconfig  # noqa: E402

# ---- Layer 1: FormatCode safe wrapper ---------------------------------------
try:
    import yapf.yapflib.yapf_api as _yapf_api  # noqa: E402

    if not getattr(_yapf_api.FormatCode, "_yapf_safe", False):
        _orig_format_code = _yapf_api.FormatCode

        def _safe_format_code(text, *args, **kwargs):
            try:
                return _orig_format_code(text, *args, **kwargs)
            except BaseException as e:  # ParseError sometimes isn't an Exception
                print(
                    f"[yapf.FormatCode] failed ({type(e).__name__}); "
                    f"returning text unformatted.",
                    flush=True,
                )
                return text, False

        _safe_format_code._yapf_safe = True
        _yapf_api.FormatCode = _safe_format_code
        # Pointcept imported FormatCode by name; rebind the symbol there too.
        _ptconfig.FormatCode = _safe_format_code
except ImportError:
    pass

# ---- Layer 2: Config.dump safe wrapper --------------------------------------
if not getattr(_ptconfig.Config.dump, "_yapf_safe", False):
    _orig_dump = _ptconfig.Config.dump

    def _safe_dump(self, file=None):
        try:
            return _orig_dump(self, file)
        except BaseException as e:
            print(
                f"[Config.dump] failed ({type(e).__name__}: "
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
