"""Render a design to a PIL image in memory, for the judge.

Reuses the vendored renderer exactly as `engine/main.py` does for the final
PNG (`renderer.converter.convert`: `_get_renderer` + `convert_chameleon_to_crello`
+ `CrelloV5Renderer.render`), but at a reduced `max_size` for speed and
without touching the filesystem. The scorer samples the pixels under text
boxes for contrast / busy-background checks, so `with_text=False` renders
everything except the text (what the text sits on).

The judge must never depend on the renderer succeeding: `engine/main.py`
writes the output JSON only after `resize()` returns, so a renderer crash
inside the loop would lose the whole task. Pixel checks are therefore a
bonus that is switched off whenever they are not known to be safe:

  * policy      off when the process looks emulated (the reported CPU
                family differs from the host's in /proc/cpuinfo: skia
                segfaults under QEMU, see docs/DOCKER_CHECK.md), or when
                CHAMELEON_RENDER=0 / CHAMELEON_NO_RENDER=1; CHAMELEON_RENDER=1
                forces it on
  * budget      a single render slower than PER_RENDER_BUDGET_S switches
                rendering off for the rest of the process; a task may spend
                at most TASK_RENDER_BUDGET_S on rendering in total
  * failures    any exception returns None; MAX_FAILURES of them switch
                rendering off for good

Without pixels the scorer's contrast terms are neutral and everything else
(hard fails, readability, layout, faithfulness) still ranks the candidates.
Nothing here writes files or modifies asset pixels; no threads, no
subprocesses, no extra native code.
"""

from __future__ import annotations

import io
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "brand_kit" / "assets"
FONTS_DIR = REPO_ROOT / "brand_kit" / "fonts"

DEFAULT_MAX_SIZE = 800
PER_RENDER_BUDGET_S = 2.5      # one render slower than this: pixels off for the process
TASK_RENDER_BUDGET_S = 6.0     # total rendering time allowed per resize() call
MAX_FAILURES = 2

_state: dict[str, Any] = {
    "checked": False,          # renderer import attempted
    "ok": False,               # renderer import succeeded
    "renderer": None,
    "convert": None,
    "policy": None,            # (allowed: bool, reason: str), decided once
    "disabled": None,          # reason rendering was switched off at run time
    "failures": 0,
    "task_spent": 0.0,
    "task_warned": False,
}


# ---------------------------------------------------------------------------
# policy


def _reported_family() -> str | None:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64", "i386", "i686", "x86"):
        return "x86"
    if m.startswith(("arm", "aarch")):
        return "arm"
    return None


def _host_family() -> str | None:
    """CPU family of the kernel's own CPU, from /proc/cpuinfo (Linux only).
    Under user-mode emulation (QEMU, Rosetta) this is the real host, while
    platform.machine() reports the emulated one."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
            head = f.read(65536)
    except OSError:
        return None
    keys = {line.split(":", 1)[0].strip().lower() for line in head.splitlines() if ":" in line}
    if keys & {"vendor_id", "flags", "model name"}:
        return "x86"
    if keys & {"cpu implementer", "cpu architecture", "features"}:
        return "arm"
    return None


def emulated() -> bool:
    reported, host = _reported_family(), _host_family()
    return reported is not None and host is not None and reported != host


def policy() -> tuple[bool, str]:
    """(allowed, reason) for pixel checks in this process; decided once."""
    if _state["policy"] is None:
        forced = os.environ.get("CHAMELEON_RENDER", "").strip().lower()
        if os.environ.get("CHAMELEON_NO_RENDER", "").strip().lower() in ("1", "true", "yes"):
            forced = "0"
        if forced in ("1", "true", "yes", "on"):
            _state["policy"] = (True, "forced on by CHAMELEON_RENDER")
        elif forced in ("0", "false", "no", "off"):
            _state["policy"] = (False, "switched off by environment")
        elif emulated():
            _state["policy"] = (False, "process looks emulated (reported %s CPU on a %s host); "
                                       "skia may crash under emulation" % (_reported_family(), _host_family()))
        else:
            _state["policy"] = (True, "native")
        if not _state["policy"][0]:
            logger.warning("judge pixel checks off: %s", _state["policy"][1])
    return _state["policy"]


def available() -> bool:
    """True when the vendored renderer imports and the brand fonts load."""
    _ensure()
    return bool(_state["ok"])


def enabled() -> bool:
    """Will render_design() try to render right now?"""
    if not policy()[0] or _state["disabled"]:
        return False
    return available()


def status() -> str:
    if not policy()[0]:
        return "off: " + policy()[1]
    if _state["disabled"]:
        return "off: " + _state["disabled"]
    if _state["checked"] and not _state["ok"]:
        return "off: renderer unavailable"
    return "on"


def begin_task() -> None:
    """Reset the per-task rendering budget (resize() calls this)."""
    _state["task_spent"] = 0.0
    _state["task_warned"] = False


def _disable(reason: str) -> None:
    if not _state["disabled"]:
        _state["disabled"] = reason
        logger.warning("judge pixel checks switched off: %s", reason)


def _ensure() -> None:
    if _state["checked"]:
        return
    _state["checked"] = True
    try:
        from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello

        _state["renderer"] = _get_renderer(str(FONTS_DIR))
        _state["convert"] = convert_chameleon_to_crello
        _state["ok"] = True
    except Exception as exc:  # pragma: no cover - only without the renderer
        logger.warning("in-memory renderer unavailable (%s); judging without pixels", exc)
        _state["ok"] = False


# ---------------------------------------------------------------------------
# rendering


def render_design(design: dict[str, Any], max_size: int = DEFAULT_MAX_SIZE,
                  with_text: bool = True, assets_dir: Path | str | None = None):
    """Render `design` ({"canvas", "elements"}) to an RGB PIL image whose
    longer side is at most `max_size` px, or None when pixel checks are off,
    the budget is spent, or rendering fails. The image is scaled uniformly:
    pixel = design px * (image.width / canvas.width)."""
    if not enabled():
        return None
    if _state["task_spent"] >= TASK_RENDER_BUDGET_S:
        if not _state["task_warned"]:
            _state["task_warned"] = True
            logger.warning("render budget for this task spent (%.1fs); remaining candidates judged without pixels",
                           _state["task_spent"])
        return None
    t0 = time.perf_counter()
    try:
        from PIL import Image

        renderer = _state["renderer"]
        example, _warnings = _state["convert"](design, renderer.font_manager,
                                               str(assets_dir or ASSETS_DIR))
        png = renderer.render(example, max_size=max_size, render_text=with_text, format="png")
        img = Image.open(io.BytesIO(png))
        img.load()
        return img.convert("RGB")
    except Exception as exc:
        _state["failures"] += 1
        logger.warning("render failed (%s); judging this candidate without pixels", exc)
        if _state["failures"] >= MAX_FAILURES:
            _disable("renderer raised %d times" % _state["failures"])
        return None
    finally:
        dt = time.perf_counter() - t0
        _state["task_spent"] += dt
        if dt > PER_RENDER_BUDGET_S:
            _disable("a render took %.1fs (budget %.1fs)" % (dt, PER_RENDER_BUDGET_S))


def image_scale(design: dict[str, Any], img) -> float:
    """Design px -> image px factor for an image returned by render_design."""
    return img.width / float(design["canvas"]["width"])


def _reset_for_tests() -> None:
    """Forget run-time decisions (policy, budget, failures); keep the import."""
    _state.update({"policy": None, "disabled": None, "failures": 0, "task_spent": 0.0, "task_warned": False})
