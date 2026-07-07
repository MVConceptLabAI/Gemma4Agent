from __future__ import annotations

"""File-entry wrapper for Unsloth's DiffusionGemma shim.

Unsloth Studio executes UNSLOTH_DG_SHIM as `python <shim_file> ...`.
The installed shim uses package-relative imports (`from . import visual_engine`), so direct
file execution breaks. This wrapper executes the installed shim with a minimal package
context for `unsloth_zoo.diffusion_studio` without importing unsloth_zoo.__init__.
"""

from pathlib import Path
import os
import sys
import types


def _rewrite_gpu_arg_for_visual_engine() -> None:
    """Preserve the runner's physical CUDA device selection.

    Unsloth Studio launches the shim with `--gpu 0` after the outer runner sets
    CUDA_VISIBLE_DEVICES=<physical nvidia-smi index>. The installed visual_engine
    then sets the child visual-server's CUDA_VISIBLE_DEVICES to that `--gpu` value,
    accidentally turning the request back into physical GPU 0 on Windows.

    Rewrite only the common Studio case (`--gpu 0`) to the already-requested
    physical CUDA_VISIBLE_DEVICES value so the visual-server loads on the target
    card (for example A100 at nvidia-smi index 1).
    """
    visible = os.environ.get("DG_PHYSICAL_CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return
    try:
        idx = sys.argv.index("--gpu")
    except ValueError:
        sys.argv.extend(["--gpu", visible])
        return
    if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in {"0", "cuda:0", "CUDA0"}:
        sys.argv[idx + 1] = visible


_rewrite_gpu_arg_for_visual_engine()

installed = Path.home() / ".unsloth" / "studio" / "unsloth_studio" / "Lib" / "site-packages"
pkg_root = installed / "unsloth_zoo"
diffusion_pkg = pkg_root / "diffusion_studio"
shim = diffusion_pkg / "shim.py"
if not shim.is_file():
    raise SystemExit(f"DiffusionGemma shim not found: {shim}")
if not (diffusion_pkg / "visual_engine.py").is_file():
    raise SystemExit(f"DiffusionGemma visual_engine not found beside shim: {diffusion_pkg}")

if str(installed) not in sys.path:
    sys.path.insert(0, str(installed))

# Avoid importing unsloth_zoo.__init__ (heavy accelerator checks) while still letting
# `from . import visual_engine` resolve from the real diffusion_studio directory.
zoo_module = sys.modules.get("unsloth_zoo")
if zoo_module is None:
    zoo_module = types.ModuleType("unsloth_zoo")
    zoo_module.__path__ = [str(pkg_root)]
    zoo_module.__package__ = "unsloth_zoo"
    sys.modules["unsloth_zoo"] = zoo_module

diffusion_module = sys.modules.get("unsloth_zoo.diffusion_studio")
if diffusion_module is None:
    diffusion_module = types.ModuleType("unsloth_zoo.diffusion_studio")
    diffusion_module.__path__ = [str(diffusion_pkg)]
    diffusion_module.__package__ = "unsloth_zoo.diffusion_studio"
    sys.modules["unsloth_zoo.diffusion_studio"] = diffusion_module
setattr(zoo_module, "diffusion_studio", diffusion_module)

globals_dict = {
    "__name__": "__main__",
    "__file__": str(shim),
    "__package__": "unsloth_zoo.diffusion_studio",
    "__builtins__": __builtins__,
}
code = compile(shim.read_text(encoding="utf-8"), str(shim), "exec")
exec(code, globals_dict)
