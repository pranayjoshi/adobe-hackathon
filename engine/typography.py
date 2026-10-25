"""Renderer-compatible text measurement.

Uses the exact font lookup, glyph widths and greedy wrap the official
renderer uses, so a fit decided here is the fit that gets rendered. The
renderer truncates font sizes to integers, so every size chosen here is an int.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
import json
from typing import Any

from .canonical import FONTS_DIR

_BOLD = {"bold", "semibold", "extrabold", "black"}

try:  # pragma: no cover - exercised implicitly
    import skia  # type: ignore

    from renderer.converter.brand_fonts import load_brand_font_manager
    from renderer.converter.wrap import wrap_text

    _FM = load_brand_font_manager(str(FONTS_DIR))
except Exception:  # renderer unavailable: fall back to an estimate
    skia = None
    _FM = None

    def wrap_text(text: str, widths: list[float], max_width: float) -> str:  # type: ignore[no-redef]
        out, line, last_space = list(text), 0.0, None
        for i, c in enumerate(out):
            if c == "\n":
                line, last_space = 0.0, None
                continue
            w = widths[i]
            if line + w > max_width and last_space is not None:
                out[last_space] = "\n"
                line = sum(widths[j] for j in range(last_space + 1, i + 1))
                last_space = None
            else:
                line += w
            if c == " ":
                last_space = i
        return "".join(out)


def _style_key(el: dict[str, Any]) -> tuple[str, str]:
    weight = (el.get("font_weight") or "regular").lower()
    bold = weight in _BOLD
    italic = (el.get("font_style") or "normal") == "italic" or el.get("italic") is True
    if bold and italic:
        style = "bolditalic"
    elif bold:
        style = "bold"
    elif italic:
        style = "italic"
    else:
        style = "regular"
    return ("bold" if bold else "regular"), style


@lru_cache(maxsize=4096)
def _typeface(family: str, weight: str, style: str):
    return skia.Typeface.MakeFromData(_FM.lookup(family, weight, style))


@lru_cache(maxsize=8192)
def _widths(family: str, weight: str, style: str, size: int, text: str, capitalize: bool = False) -> tuple[float, ...]:
    if not text:
        return ()
    if _FM is None:
        return tuple(size * (0.32 if c == " " else 0.56) for c in text)
    font = skia.Font(_typeface(family, weight, style), size)
    if capitalize:
        return tuple(sum(font.getWidths(font.textToGlyphs(c.upper()))) for c in text)
    return tuple(font.getWidths(font.textToGlyphs(text)))


@lru_cache(maxsize=2048)
def _metrics(family: str, weight: str, style: str, size: int) -> tuple[float, float, float]:
    """(ascent [negative], descent, cap height) in px."""
    if _FM is None:
        return -0.92 * size, 0.24 * size, 0.7 * size
    m = skia.Font(_typeface(family, weight, style), size).getMetrics()
    cap = m.fCapHeight if m.fCapHeight > 0 else 0.7 * size
    return m.fAscent, m.fDescent, cap


@dataclass(frozen=True)
class TextFit:
    font_size: int
    box_width: float
    height: float          # full line-box height (top of first ascent to last descent)
    lines: int
    line_width: float      # widest rendered line
    overflow: bool         # an unbreakable word is wider than the box
    visual_center: float   # offset from box top to the optical centre of the glyphs


def measure(el: dict[str, Any], font_size: int, box_width: float) -> TextFit:
    font_size = max(1, int(font_size))
    content = str(el.get("content", ""))
    family = el.get("font_family", "")
    weight, style = _style_key(el)
    widths = _widths(family, weight, style, font_size, content, bool(el.get("capitalize",False)))
    wrapped = wrap_text(content, list(widths), float(box_width)) if content else ""

    # Wrapping uses the base face in the official converter. The rendered
    # per-character face can still be wider, so measure it after wrapping.
    widths=list(widths)
    # The native renderer starts a new blob at every style/color boundary;
    # letter spacing applies inside each blob rather than across its boundary.
    styles = [(weight, style, el.get("color", "rgba(0,0,0,1)")) for _ in content]
    for run in el.get("style_runs") or []:
        for i in range(max(0,int(run.get("start",0))),min(len(content),int(run.get("end",len(content)-1))+1)):
            old_weight, old_style, old_color = styles[i]
            styled = {"font_weight": old_weight,
                      "italic": old_style in ("italic", "bolditalic")}
            if "font_weight" in run:
                styled["font_weight"] = run["font_weight"]
            if "font_style" in run or "italic" in run:
                styled["italic"] = (run.get("font_style") == "italic" or run.get("italic") is True)
            rw, rs = _style_key(styled)
            styles[i] = (rw, rs, run.get("color", old_color))
    for i, (rw, rs, _) in enumerate(styles):
        widths[i] = _widths(family, rw, rs, font_size, content, bool(el.get("capitalize", False)))[i]
    line_widths, cur = [], 0.0
    spacing = float(el.get("letter_spacing") or 0.0)
    previous = None
    for i, c in enumerate(wrapped):
        if c == "\n":
            line_widths.append(cur)
            cur = 0.0
            previous = None
        else:
            cur += widths[i] if i < len(widths) else 0.0
            glyph_count = len(c.upper()) if el.get("capitalize") else 1
            cur += spacing * max(0, glyph_count - (0 if previous == styles[i] else 1))
            previous = styles[i]
    line_widths.append(cur)
    # The renderer draws trailing spaces too, but they are invisible.
    widest = max(line_widths) if line_widths else 0.0

    n = len(line_widths)
    lh = float(el.get("line_height") or 1.2)
    asc, desc, cap = _metrics(family, weight, style, font_size)
    step = font_size * lh
    height = (n - 1) * step + (desc - asc)
    cap_top = -asc - cap
    last_baseline = (n - 1) * step - asc
    return TextFit(
        font_size=font_size,
        box_width=float(box_width),
        height=height,
        lines=n,
        line_width=widest,
        overflow=widest > box_width + 0.5,
        visual_center=(cap_top + last_baseline) / 2,
    )


def tight_width(el: dict[str, Any], fit: TextFit) -> float:
    """Narrowest box width that reproduces the same wrap as `fit`.

    Greedy wrap counts a line's trailing space before breaking, so pad by
    one space width and verify the line count is unchanged."""
    if fit.overflow:
        return fit.box_width
    family = el.get("font_family", "")
    weight, style = _style_key(el)
    space = (_widths(family, weight, style, fit.font_size, " ") or (fit.font_size * 0.3,))[0]
    w = min(fit.box_width, fit.line_width + space + 2)
    if measure(el, fit.font_size, w).lines != fit.lines:
        return fit.box_width
    return w


def fit_text(
    el: dict[str, Any],
    box_width: float,
    max_height: float,
    fs_max: int,
    fs_min: int = 1,
) -> TextFit:
    """Largest integer font size in [fs_min, fs_max] that wraps inside
    `box_width` without overflow and fits `max_height`."""
    fs_max = max(1, int(fs_max))
    fs_min = max(1, min(int(fs_min), fs_max))

    def ok(fs: int) -> bool:
        f = measure(el, fs, box_width)
        return not f.overflow and f.height <= max_height + 0.5

    if ok(fs_max):
        return measure(el, fs_max, box_width)
    lo, hi = fs_min, fs_max  # invariant: hi fails
    if not ok(lo):
        return measure(el, lo, box_width)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return measure(el, lo, box_width)


def source_lines(el: dict[str, Any]) -> int:
    return measure(el, int(el.get("font_size", 12)), float(el["width"])).lines


def renderer_available() -> bool:
    return _FM is not None


@lru_cache(maxsize=1024)
def _face_identity(family: str, weight: str, style: str) -> tuple[str, bool]:
    """Fingerprint the actual face; a renderer fallback is never verified.

    Fonts are immutable within a worker, just like the renderer's font manager.
    Reload the worker when changing font resources.
    """
    if _FM is None:
        return "estimated", False
    from cr_renderer.fonts import normalize_family, get_font_weight, get_font_style
    faces = _FM._fonts.get(normalize_family(family), [])
    exact = any(f.get("fontWeight", "regular") == get_font_weight(family, weight)
                and f.get("fontStyle", "regular") == get_font_style(family, style)
                for f in faces)
    data = _FM.lookup(family, weight, style)
    return hashlib.sha256(data).hexdigest(), exact


@lru_cache(maxsize=1)
def _measurement_profile() -> str:
    from importlib.metadata import version, PackageNotFoundError
    try:
        runtime = version("skia-python")
    except PackageNotFoundError:
        runtime = "unavailable"
    return hashlib.sha256((runtime + inspect.getsource(measure) + inspect.getsource(wrap_text)
                           + inspect.getsource(_widths) + inspect.getsource(_metrics)).encode()).hexdigest()


def measurement_identity(el: dict[str, Any]) -> dict[str, Any]:
    """Resource/style identity for measured configuration and training records.

    ``verified`` refers to exact local font availability and compatible metrics,
    not a claim that a pixel-level render has been performed.
    """
    variants = [el] + [{**el, **run} for run in el.get("style_runs") or []]
    faces = [_face_identity(str(el.get("font_family", "")), *_style_key(v)) for v in variants]
    typography = {k: el.get(k) for k in (
        "content", "font_family", "font_weight", "font_style", "italic",
        "capitalize", "letter_spacing", "line_height", "style_runs")}
    profile = _measurement_profile()
    fingerprint = hashlib.sha256(json.dumps({"typography": typography, "faces": faces,
                                            "profile": profile}, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": fingerprint, "profile": profile,
            "renderer_mode": "official_metrics" if _FM is not None else "estimated",
            "verified": _FM is not None and all(exact for _, exact in faces),
            "font_hashes": tuple(face for face, _ in faces)}
