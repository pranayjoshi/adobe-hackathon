"""Per-element feature vectors for the learned role classifier.

This module reproduces (and extends) the signals `engine.analyze` reads out of
a design, as plain numbers, without touching the analyzer: type, area share,
position, aspect, opacity, z rank, font-size rank, weight, letter spacing,
text length / word count / case / digits / dates / verbs, grouping, what sits
behind or next to an element, counts over the whole design, canvas aspect.

Everything is derived from id-free fields only (type, geometry, text
properties, colours): hidden tasks give opaque ids. The feature order is fixed
by `FEATURE_NAMES`; `engine/models/roles.json.gz` records the names it was
trained with and `engine.role_model` refuses a model whose names differ.

Pure Python, deterministic, O(n^2) in the number of elements (n <= ~50).
"""

from __future__ import annotations

import math
import re
from typing import Any

from engine.geometry import Box

# Short imperative openers typical of a call to action. A tiny list on purpose:
# the classifier only needs a hint, the rest comes from geometry and weight.
_VERBS = {
    "shop", "buy", "get", "order", "book", "join", "sign", "register", "learn", "discover", "try",
    "start", "visit", "call", "download", "subscribe", "explore", "find", "read", "watch", "grab",
    "claim", "save", "enter", "apply", "contact", "reserve", "donate", "see", "swipe", "tap", "click",
    "follow", "enjoy", "taste", "meet", "win", "play", "listen", "rsvp", "vote", "attend", "come",
}
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_DATE_RE = re.compile(
    r"(\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b)|(\b\d{1,2}(st|nd|rd|th)?\s+(" + "|".join(_MONTHS)
    + r")[a-z]*\b)|(\b(" + "|".join(_MONTHS) + r")[a-z]*\.?\s+\d{1,2}\b)|(\b(19|20)\d{2}\b)",
    re.I,
)
_TIME_RE = re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b|\b\d{1,2}:\d{2}\b", re.I)
_URL_RE = re.compile(r"(https?://|www\.|\.com\b|\.org\b|\.net\b|\.co\b|\.io\b|@[a-z0-9_]+)", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
_PRICE_RE = re.compile(r"([$€£¥]\s?\d)|(\d\s?%)|(\b\d+\s?(usd|eur|gbp|off)\b)", re.I)
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ']+")


# ---------------------------------------------------------------------------
# small helpers (duplicated from analyze on purpose: this module must not
# depend on the analyzer's thresholds, which other work keeps changing)


def _rgba(color: Any) -> tuple[float, float, float, float]:
    if not isinstance(color, str) or "(" not in color:
        return (0.0, 0.0, 0.0, 1.0)
    inner = color[color.find("(") + 1: color.rfind(")")]
    parts = [p.strip() for p in inner.split(",")]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return (0.0, 0.0, 0.0, 1.0)
    if len(vals) < 3:
        return (0.0, 0.0, 0.0, 1.0)
    a = vals[3] if len(vals) > 3 else 1.0
    return (vals[0], vals[1], vals[2], max(0.0, min(1.0, a)))


def _luminance(rgb: tuple[float, float, float]) -> float:
    def ch(c: float) -> float:
        c = max(0.0, min(255.0, c)) / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _contrast(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _saturation(rgb: tuple[float, float, float]) -> float:
    r, g, b = (c / 255.0 for c in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    if mx <= 0:
        return 0.0
    return (mx - mn) / mx


def _log(v: float) -> float:
    return math.log(max(v, 1e-6))


def _gap(a: Box, b: Box) -> float:
    """Euclidean gap between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a.x, b.x) - min(a.right, b.right))
    dy = max(0.0, max(a.y, b.y) - min(a.bottom, b.bottom))
    return math.hypot(dx, dy)


def _text_stats(content: str) -> dict[str, float]:
    words = _WORD_RE.findall(content)
    letters = [c for c in content if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    digits = sum(1 for c in content if c.isdigit())
    stripped = content.strip()
    first = words[0].lower() if words else ""
    lines = [ln for ln in content.split("\n") if ln.strip()]
    return {
        "n_chars": float(len(stripped)),
        "n_words": float(len(words)),
        "n_lines": float(max(1, len(lines))),
        "longest_line": float(max((len(ln) for ln in lines), default=0)),
        "upper_ratio": upper / len(letters) if letters else 0.0,
        "digit_ratio": digits / len(stripped) if stripped else 0.0,
        "has_digits": 1.0 if digits else 0.0,
        "has_date": 1.0 if _DATE_RE.search(content) else 0.0,
        "has_time": 1.0 if _TIME_RE.search(content) else 0.0,
        "has_url": 1.0 if (_URL_RE.search(content) or _EMAIL_RE.search(content)) else 0.0,
        "has_phone": 1.0 if _PHONE_RE.search(content) else 0.0,
        "has_price": 1.0 if _PRICE_RE.search(content) else 0.0,
        "verb_start": 1.0 if first in _VERBS else 0.0,
        "ends_excl": 1.0 if stripped.endswith("!") else 0.0,
        "ends_period": 1.0 if stripped.endswith((".", ";")) else 0.0,
        "has_comma": 1.0 if "," in stripped else 0.0,
        "avg_word_len": (sum(len(w) for w in words) / len(words)) if words else 0.0,
        "dots_only": 1.0 if stripped and all(not c.isalnum() for c in stripped) else 0.0,
    }


# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    # identity
    "is_text", "is_image", "is_shape", "shape_rect", "shape_ellipse", "shape_line", "fit_cover",
    # canvas / design context
    "canvas_log_aspect", "canvas_log_area", "n_elements", "n_texts", "n_images", "n_shapes", "n_groups",
    # geometry
    "area_share", "visible_share", "w_share", "h_share", "log_aspect", "cx", "cy", "left", "top",
    "right_margin", "bottom_margin", "inside_frac", "min_side_share", "spans_x", "spans_y",
    "dist_center", "opacity", "rotation_abs", "z_rank", "z_top", "z_bottom",
    # grouping
    "grouped", "group_size", "group_has_text", "group_has_shape", "group_has_image",
    "group_area_share", "group_text_share",
    # neighbourhood
    "n_overlap", "n_overlap_text", "n_above_z_inside", "behind_shape", "behind_shape_area",
    "behind_shape_ratio", "behind_image", "backs_text", "backs_n_text", "backs_text_ratio",
    "backs_text_chars", "backs_text_fontpos", "dist_text", "dist_image", "dist_shape",
    "largest_image", "image_area_rank", "image_area_ratio", "n_bigger_images",
    "reading_pos", "text_reading_pos",
    # text properties
    "font_px", "font_rel", "font_ratio_max", "font_rank", "font_rank_norm", "font_ratio_next",
    "font_ratio_prev", "font_same_count", "font_ratio_median", "bold", "italic", "letter_sp_ratio",
    "line_height", "align_left", "align_center", "align_right", "has_runs",
    "n_chars", "n_words", "n_lines", "longest_line", "upper_ratio", "digit_ratio", "has_digits",
    "has_date", "has_time", "has_url", "has_phone", "has_price", "verb_start", "ends_excl",
    "ends_period", "has_comma", "avg_word_len", "dots_only",
    "chars_per_line_cap", "est_lines", "box_lines", "ink_share", "text_fill",
    "lum", "alpha", "sat", "contrast_bg", "contrast_behind", "color_shared", "color_unique",
    "text_above_ratio", "text_below_ratio", "first_text", "last_text", "max_text_y_rank",
    # shape / image properties
    "fill_lum", "fill_sat", "corner_ratio", "thin", "covers_canvas", "same_fill_count",
    "is_biggest_text", "text_area_rank",
]
N_FEATURES = len(FEATURE_NAMES)
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def features(source: dict[str, Any]) -> tuple[list[str], list[list[float]]]:
    """Return (element ids, rows) with one row of `N_FEATURES` floats per element,
    in the order of `source["elements"]`."""
    canvas = source.get("canvas") or {}
    W = max(1.0, float(canvas.get("width", 1) or 1))
    H = max(1.0, float(canvas.get("height", 1) or 1))
    cbox = Box(0.0, 0.0, W, H)
    carea = W * H
    diag = math.hypot(W, H)
    elements = list(source.get("elements") or [])
    n = len(elements)
    ids = [str(el.get("id", i)) for i, el in enumerate(elements)]
    boxes = [Box.of(el) for el in elements]
    types = [str(el.get("type", "")) for el in elements]
    zs = [float(el.get("z_index", i)) if el.get("z_index") is not None else float(i)
          for i, el in enumerate(elements)]
    z_order = sorted(range(n), key=lambda i: (zs[i], i))
    z_rank = {i: r for r, i in enumerate(z_order)}

    texts = [i for i in range(n) if types[i] == "text"]
    images = [i for i in range(n) if types[i] == "image"]
    shapes = [i for i in range(n) if types[i] == "shape"]

    groups: dict[str, list[int]] = {}
    for i, el in enumerate(elements):
        gid = el.get("group_id")
        if gid:
            groups.setdefault(str(gid), []).append(i)

    # --- design-level text statistics
    fonts = {i: float(elements[i].get("font_size", 0.0) or 0.0) for i in texts}
    font_sorted = sorted(texts, key=lambda i: (-fonts[i], i))
    font_rank = {i: r for r, i in enumerate(font_sorted)}
    max_font = max(fonts.values(), default=1.0) or 1.0
    median_font = sorted(fonts.values())[len(fonts) // 2] if fonts else 1.0
    text_by_y = sorted(texts, key=lambda i: (boxes[i].cy, boxes[i].cx))
    text_y_rank = {i: r for r, i in enumerate(text_by_y)}
    text_area_sorted = sorted(texts, key=lambda i: (-boxes[i].area, i))
    text_area_rank = {i: r for r, i in enumerate(text_area_sorted)}
    colors = {i: _rgba(elements[i].get("color")) for i in texts}
    color_count: dict[tuple[float, float, float], int] = {}
    for i in texts:
        key = colors[i][:3]
        color_count[key] = color_count.get(key, 0) + 1

    # --- images
    image_sorted = sorted(images, key=lambda i: (-boxes[i].area, i))
    image_rank = {i: r for r, i in enumerate(image_sorted)}
    max_image_area = boxes[image_sorted[0]].area if image_sorted else 0.0

    # --- shapes
    fills = {i: _rgba(elements[i].get("fill_color")) for i in shapes}
    fill_count: dict[tuple[float, float, float], int] = {}
    for i in shapes:
        key = fills[i][:3]
        fill_count[key] = fill_count.get(key, 0) + 1

    # --- the dominant background colour: the lowest-z element covering most of
    # the canvas; shapes give their fill, images count as "unknown" (mid grey).
    bg_rgb: tuple[float, float, float] = (255.0, 255.0, 255.0)
    best_cover = 0.0
    for i in z_order:
        if types[i] == "text":
            continue
        cover = cbox.intersection_area(boxes[i]) / carea
        if cover >= 0.5 and cover >= best_cover - 1e-9 and best_cover == 0.0:
            best_cover = cover
            if types[i] == "shape":
                bg_rgb = fills[i][:3]
            else:
                bg_rgb = (128.0, 128.0, 128.0)
            break

    # reading order over everything but full-bleed elements
    content_idx = [i for i in range(n) if cbox.intersection_area(boxes[i]) / carea < 0.85]
    reading = sorted(content_idx, key=lambda i: (round(boxes[i].cy / max(H, 1.0), 2), boxes[i].cx))
    reading_pos = {i: r for r, i in enumerate(reading)}

    rows: list[list[float]] = []
    for i, el in enumerate(elements):
        f = [0.0] * N_FEATURES

        def put(name: str, value: float) -> None:
            f[_INDEX[name]] = float(value)

        b = boxes[i]
        t = types[i]
        area = b.area
        put("is_text", t == "text")
        put("is_image", t == "image")
        put("is_shape", t == "shape")
        st = str(el.get("shape_type", "") or "")
        put("shape_rect", st == "rectangle")
        put("shape_ellipse", st == "ellipse")
        put("shape_line", st == "line")
        put("fit_cover", str(el.get("fit", "contain") or "contain") == "cover")

        put("canvas_log_aspect", _log(W / H))
        put("canvas_log_area", _log(carea) - _log(1080.0 * 1080.0))
        put("n_elements", n)
        put("n_texts", len(texts))
        put("n_images", len(images))
        put("n_shapes", len(shapes))
        put("n_groups", len(groups))

        visible = cbox.intersection_area(b)
        put("area_share", min(4.0, area / carea))
        put("visible_share", visible / carea)
        put("w_share", min(4.0, b.w / W))
        put("h_share", min(4.0, b.h / H))
        put("log_aspect", _log(max(b.w, 1e-3) / max(b.h, 1e-3)))
        put("cx", b.cx / W)
        put("cy", b.cy / H)
        put("left", b.x / W)
        put("top", b.y / H)
        put("right_margin", (W - b.right) / W)
        put("bottom_margin", (H - b.bottom) / H)
        put("inside_frac", cbox.contains_fraction(b) if area > 0 else 1.0)
        put("min_side_share", min(b.w / W, b.h / H))
        put("spans_x", b.x <= 0.02 * W and b.right >= 0.98 * W)
        put("spans_y", b.y <= 0.02 * H and b.bottom >= 0.98 * H)
        put("dist_center", math.hypot((b.cx - W / 2) / W, (b.cy - H / 2) / H))
        put("opacity", float(el.get("opacity", 1.0) if el.get("opacity") is not None else 1.0))
        put("rotation_abs", min(180.0, abs(float(el.get("rotation", 0.0) or 0.0))) / 180.0)
        put("z_rank", z_rank[i] / max(1, n - 1))
        put("z_top", z_rank[i] == n - 1)
        put("z_bottom", z_rank[i] == 0)

        gid = el.get("group_id")
        members = groups.get(str(gid), []) if gid else []
        put("grouped", bool(gid))
        put("group_size", len(members))
        if members:
            mtypes = {types[j] for j in members}
            put("group_has_text", "text" in mtypes)
            put("group_has_shape", "shape" in mtypes)
            put("group_has_image", "image" in mtypes)
            gbox = Box.union(boxes[j] for j in members)
            put("group_area_share", min(4.0, gbox.area / carea))
            tarea = sum(boxes[j].area for j in members if types[j] == "text")
            put("group_text_share", tarea / gbox.area if gbox.area > 0 else 0.0)

        # neighbourhood
        n_overlap = n_overlap_text = n_above_inside = 0
        behind_shape = -1
        behind_shape_cover = 0.0
        behind_image = 0.0
        backs: list[int] = []
        d_text = d_image = d_shape = 2.0
        for j in range(n):
            if j == i:
                continue
            o = boxes[j]
            if o.area <= 0 or area <= 0:
                continue
            if b.overlaps(o):
                n_overlap += 1
                if types[j] == "text":
                    n_overlap_text += 1
            inside_me = b.contains_fraction(o)
            me_inside = o.contains_fraction(b)
            if zs[j] > zs[i] and inside_me >= 0.5:
                n_above_inside += 1
            if zs[j] < zs[i] and me_inside >= 0.8 and o.area < 0.85 * carea + 1e-9 \
                    and types[j] != "text":
                if types[j] == "shape":
                    # the closest (smallest) shape behind this element
                    if behind_shape < 0 or o.area < boxes[behind_shape].area:
                        behind_shape = j
                        behind_shape_cover = me_inside
                else:
                    behind_image = 1.0
            if t != "text" and types[j] == "text" and zs[j] > zs[i] and inside_me >= 0.8 \
                    and area < 0.85 * carea:
                backs.append(j)
            g = _gap(b, o) / diag
            if types[j] == "text":
                d_text = min(d_text, g)
            elif types[j] == "image":
                d_image = min(d_image, g)
            else:
                d_shape = min(d_shape, g)
        put("n_overlap", n_overlap)
        put("n_overlap_text", n_overlap_text)
        put("n_above_z_inside", n_above_inside)
        if behind_shape >= 0:
            sb = boxes[behind_shape]
            put("behind_shape", 1.0)
            put("behind_shape_area", min(4.0, sb.area / carea))
            put("behind_shape_ratio", area / sb.area if sb.area > 0 else 0.0)
        put("behind_image", behind_image)
        if backs:
            put("backs_text", 1.0)
            put("backs_n_text", len(backs))
            tarea = sum(boxes[j].area for j in backs)
            put("backs_text_ratio", min(4.0, tarea / area) if area > 0 else 0.0)
            put("backs_text_chars", sum(len(str(elements[j].get("content", ""))) for j in backs))
            best = min(backs, key=lambda j: font_rank.get(j, 99))
            put("backs_text_fontpos", 1.0 - font_rank.get(best, 0) / max(1, len(texts) - 1)
                if len(texts) > 1 else 1.0)
        put("dist_text", d_text)
        put("dist_image", d_image)
        put("dist_shape", d_shape)

        if t == "image":
            put("largest_image", image_rank[i] == 0)
            put("image_area_rank", image_rank[i])
            put("image_area_ratio", area / max_image_area if max_image_area > 0 else 0.0)
            put("n_bigger_images", image_rank[i])
        put("reading_pos", reading_pos.get(i, len(reading)) / max(1, len(reading)))

        if t == "text":
            fs = fonts[i]
            stats = _text_stats(str(el.get("content", "")))
            put("font_px", fs)
            put("font_rel", fs / math.sqrt(carea) * 100.0)
            put("font_ratio_max", fs / max_font)
            put("font_rank", font_rank[i])
            put("font_rank_norm", font_rank[i] / max(1, len(texts) - 1) if len(texts) > 1 else 0.0)
            r = font_rank[i]
            nxt = fonts[font_sorted[r + 1]] if r + 1 < len(font_sorted) else 0.0
            prv = fonts[font_sorted[r - 1]] if r > 0 else fs
            put("font_ratio_next", nxt / fs if fs > 0 else 0.0)
            put("font_ratio_prev", fs / prv if prv > 0 else 1.0)
            put("font_same_count", sum(1 for j in texts if abs(fonts[j] - fs) <= 0.05 * fs) - 1)
            put("font_ratio_median", fs / median_font if median_font > 0 else 0.0)
            put("bold", str(el.get("font_weight", "regular")) == "bold")
            put("italic", str(el.get("font_style", "normal")) == "italic")
            put("letter_sp_ratio", float(el.get("letter_spacing", 0.0) or 0.0) / fs if fs > 0 else 0.0)
            put("line_height", float(el.get("line_height", 1.2) or 1.2))
            al = str(el.get("text_align", "left") or "left")
            put("align_left", al == "left")
            put("align_center", al == "center")
            put("align_right", al == "right")
            put("has_runs", bool(el.get("style_runs")))
            for k, v in stats.items():
                put(k, v)
            est_cpl = b.w / (0.5 * fs) if fs > 0 else 0.0     # ~0.5 em per character
            put("chars_per_line_cap", min(400.0, est_cpl))
            est_lines = max(1.0, math.ceil(stats["n_chars"] / est_cpl)) if est_cpl > 0 else 1.0
            put("est_lines", min(40.0, est_lines))
            lh = float(el.get("line_height", 1.2) or 1.2)
            put("box_lines", min(40.0, b.h / (fs * lh)) if fs > 0 else 0.0)
            ink = stats["n_chars"] * 0.5 * fs * fs            # rough ink area
            put("ink_share", min(2.0, ink / carea))
            put("text_fill", min(4.0, ink / area) if area > 0 else 0.0)
            rgb, alpha = colors[i][:3], colors[i][3]
            put("lum", _luminance(rgb))
            put("alpha", alpha)
            put("sat", _saturation(rgb))
            put("contrast_bg", _contrast(rgb, bg_rgb) / 21.0)
            if behind_shape >= 0:
                put("contrast_behind", _contrast(rgb, fills[behind_shape][:3]) / 21.0)
            else:
                put("contrast_behind", _contrast(rgb, bg_rgb) / 21.0)
            put("color_shared", color_count.get(rgb, 1) - 1)
            put("color_unique", color_count.get(rgb, 1) == 1)
            yr = text_y_rank[i]
            above = text_by_y[yr - 1] if yr > 0 else None
            below = text_by_y[yr + 1] if yr + 1 < len(text_by_y) else None
            put("text_above_ratio", fonts[above] / fs if (above is not None and fs > 0) else 0.0)
            put("text_below_ratio", fonts[below] / fs if (below is not None and fs > 0) else 0.0)
            put("first_text", yr == 0)
            put("last_text", yr == len(texts) - 1)
            put("max_text_y_rank", yr / max(1, len(texts) - 1) if len(texts) > 1 else 0.0)
            put("text_reading_pos", yr / max(1, len(texts) - 1) if len(texts) > 1 else 0.0)
            put("is_biggest_text", font_rank[i] == 0)
            put("text_area_rank", text_area_rank[i])
        elif t == "shape":
            rgb = fills[i][:3]
            put("fill_lum", _luminance(rgb))
            put("fill_sat", _saturation(rgb))
            cr = float(el.get("corner_radius", 0.0) or 0.0)
            put("corner_ratio", min(1.0, cr / max(1e-6, min(b.w, b.h) / 2.0)))
            put("same_fill_count", fill_count.get(rgb, 1) - 1)
        if t != "text":
            put("thin", min(b.w, b.h) / max(b.w, b.h, 1e-6) < 0.12)
            put("covers_canvas", visible / carea)
        rows.append(f)
    return ids, rows


def feature_dict(source: dict[str, Any]) -> dict[str, dict[str, float]]:
    """{id: {feature name: value}} - handy for inspection and tests."""
    ids, rows = features(source)
    return {eid: dict(zip(FEATURE_NAMES, row)) for eid, row in zip(ids, rows)}
