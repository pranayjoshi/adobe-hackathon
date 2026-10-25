"""AnalysisGraph: a versioned, read-only sidecar describing elements and relations.

The graph sits beside the immutable design (never inside it). It records, for
each original element, its semantic role with a confidence and the source of
that role, and typed edges between elements with their provenance:

    contains           a (panel/shape/visible image) holds b inside its opaque area
    label_for          text b labels shape/image a (button, badge)
    reads_before       block containing a is read before block containing b
    aligns_with        a and b share an edge or centre line (attribute: axis)
    supports_on_photo  text b sits on photograph a (subject-bearing raster)
    overlaps           a and b intersect without containment

Nothing here uses element IDs, names or file names as evidence: IDs are only
keys. Geometry, typography, paint order, explicit `group_id` and the pixels of
the original assets (alpha coverage, colour complexity) are the only inputs.

Consumers (layout, retrieval, repair, refine, controller) should treat the graph
as optional and fall back to their legacy behaviour when it is absent.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping

from .canonical import ASSETS_DIR, Box, element_box, rotated_extent, visual_box

GRAPH_VERSION = "analysis-graph/1"
EDGE_KINDS = ("contains", "label_for", "reads_before", "aligns_with", "supports_on_photo", "overlaps")
ROLE_SOURCES = ("explicit", "model", "heuristic")
MAX_EDGES = 4000

# Heuristic role reliabilities used when no calibrated artifact supplies them.
# They are deliberately modest priors, not measured accuracies; a calibrated
# role artifact (engine/models/roles.v2.json) replaces them with weak-label
# agreement measured on held-out source families.
_PRIOR_CONFIDENCE = {
    "background": 0.9, "decoration": 0.6, "unknown": 0.3, "accent": 0.5,
    "headline": 0.7, "subhead": 0.5, "body": 0.5, "legal": 0.55, "cta": 0.6,
    "badge": 0.6, "badges": 0.6, "logo": 0.5, "hero": 0.55, "image": 0.5,
    "sticker": 0.7, "group": 0.6,
}


# ---------------------------------------------------------------------------
# Asset pixel evidence (cached by asset id; original pixels are only read)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetStats:
    exists: bool
    coverage: float = 0.0       # fraction of thumbnail pixels with alpha > 40
    colors95: int = 0           # quantised colours covering 95% of visible pixels
    edges: float = 0.0          # strong-edge density inside the opaque area
    kind: str = "missing"       # photo | graphic | linework | flat | missing


def asset_stats(asset_id: str | None) -> AssetStats:
    return _asset_stats(asset_id,_asset_signature(asset_id))


def _asset_signature(asset_id):
    path=(ASSETS_DIR/str(asset_id or '')).resolve()
    try:
        if not path.is_relative_to(ASSETS_DIR.resolve()):return ('unsafe',)
        stat=path.stat()
        return str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns
    except OSError:return ('missing',str(path))


@lru_cache(maxsize=1024)
def _asset_stats(asset_id,resource_signature) -> AssetStats:
    """Classify an asset's pixels. 'graphic' = few flat colours and few internal
    edges (a vector shape exported as raster); 'linework' = sparse visible
    pixels (rules, outlines, thin icons); 'photo' = everything else."""
    path = (ASSETS_DIR / str(asset_id or "")).resolve()
    if not asset_id or not path.is_relative_to(ASSETS_DIR.resolve()) or not path.is_file():
        return AssetStats(False)
    try:
        from PIL import Image, ImageFilter

        with Image.open(path) as im:
            im = im.convert("RGBA")
            im.thumbnail((96, 96))
            raw = im.tobytes()
            px = [tuple(raw[j:j + 4]) for j in range(0, len(raw), 4)]
            # Edges inside the shape only: erode the alpha mask so the outline
            # of a cut-out (an icon, a triangle) is not counted as texture.
            flat = Image.new("RGBA", im.size, (128, 128, 128, 255))
            flat.alpha_composite(im)
            edge = flat.convert("L").filter(ImageFilter.FIND_EDGES).tobytes()
            core = im.getchannel("A").filter(ImageFilter.MinFilter(5)).tobytes()
            tw, th = im.size
    except Exception:
        return AssetStats(False)
    visible = [p for p in px if p[3] > 40]
    coverage = len(visible) / max(1, len(px))
    if not visible:
        return AssetStats(True, 0.0, 0, 0.0, "linework")
    counts = Counter((r // 24, g // 24, b // 24) for r, g, b, _ in visible)
    acc, k = 0, 0
    for _, n in counts.most_common():
        acc += n
        k += 1
        if acc >= 0.95 * len(visible):
            break
    # Ignore the one-pixel thumbnail frame, where the edge filter has no
    # neighbours (it would make every thin rule look textured).
    inner = [edge[i] for i in range(len(core)) if core[i] > 240 and 0 < i % tw < tw - 1 and 0 < i // tw < th - 1]
    edges = sum(e > 24 for e in inner) / max(1, len(inner))
    if k <= 1 and edges < 0.02 and coverage > 0.9:
        kind = "flat"
    elif coverage < 0.12:
        kind = "linework"
    elif k <= 3 and edges <= 0.12:
        kind = "graphic"
    else:
        kind = "photo"
    return AssetStats(True, round(coverage, 4), k, round(edges, 4), kind)


def _alpha_thumb(asset_id: str):
    return _alpha_thumb_cached(asset_id,_asset_signature(asset_id))


@lru_cache(maxsize=512)
def _alpha_thumb_cached(asset_id,resource_signature):
    path = (ASSETS_DIR / str(asset_id or "")).resolve()
    if not path.is_relative_to(ASSETS_DIR.resolve()) or not path.is_file():
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            iw, ih = im.size
            a = im.convert("RGBA").getchannel("A")
            a.thumbnail((128, 128))
            return iw, ih, a.size, a.tobytes()
    except Exception:
        return None


def alpha_coverage(el: dict[str, Any], box: Box, threshold: int = 128) -> float:
    """Fraction of `box` (source canvas px) lying on opaque pixels of image `el`.

    Non-images and missing assets are treated as fully opaque over their box.
    Rotated images fall back to their box (conservative)."""
    eb = element_box(el)
    if box.area <= 0:
        return 0.0
    if el.get("type") != "image" or abs(float(el.get("rotation") or 0)) > 0.5:
        return eb.intersection_area(box) / box.area
    info = _alpha_thumb(el.get("asset_id") or "")
    if info is None:
        return eb.intersection_area(box) / box.area
    iw, ih, (tw, th), alpha = info
    fit = el.get("fit") or "contain"
    if fit == "fill":
        ox, oy, dw, dh = eb.x, eb.y, eb.w, eb.h
    else:
        k = (max if fit == "cover" else min)(eb.w / max(1, iw), eb.h / max(1, ih))
        dw, dh = iw * k, ih * k
        ox, oy = eb.x + (eb.w - dw) / 2, eb.y + (eb.h - dh) / 2
    n = hit = 0
    for j in range(6):
        for i in range(12):
            x = box.x + box.w * (i + 0.5) / 12
            y = box.y + box.h * (j + 0.5) / 6
            n += 1
            if not (eb.x <= x <= eb.x1 and eb.y <= y <= eb.y1):
                continue
            u, v = (x - ox) / max(1e-6, dw), (y - oy) / max(1e-6, dh)
            if 0 <= u < 1 and 0 <= v < 1 and alpha[min(th - 1, int(v * th)) * tw + min(tw - 1, int(u * tw))] >= threshold:
                hit += 1
    return hit / n


def glyph_box(el: dict[str, Any]) -> Box:
    """Visible type band of a text element at its source size: cap top to the
    last baseline plus a descender allowance, over the widest rendered line.
    Trailing blank lines (common in imported text boxes) are ignored."""
    from .typography import measure

    b = element_box(el)
    if el.get("type") != "text" or abs(float(el.get("rotation") or 0)) > 0.5:
        return rotated_extent(el)
    content = str(el.get("content", "")).rstrip()
    if not content:
        return Box(b.x, b.y, 0.0, 0.0)
    fs = max(1, int(el.get("font_size", 12)))
    f = measure({**el, "content": content}, fs, max(1.0, b.w))
    from .typography import _metrics, _style_key  # renderer-accurate metrics (read-only use)

    weight, style = _style_key(el)
    asc, desc, cap = _metrics(el.get("font_family", ""), weight, style, fs)
    cap_top = -asc - cap
    step = fs * float(el.get("line_height") or 1.2)
    last_baseline = (f.lines - 1) * step - asc
    top = b.y + max(0.0, cap_top)
    bottom = b.y + last_baseline + 0.6 * desc
    width = min(b.w, f.line_width) if not f.overflow else f.line_width
    align = el.get("text_align", "left")
    x = b.x + ((b.w - width) / 2 if align == "center" else (b.w - width) if align == "right" else 0.0)
    return Box(x, top, max(0.0, width), max(0.0, bottom - top))


# ---------------------------------------------------------------------------
# Graph record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisGraph:
    version: str
    nodes: Mapping[str, Mapping[str, Any]]
    edges: tuple[Mapping[str, Any], ...]
    group_policies: tuple[Any, ...]
    confidence: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def edges_of(self, kind: str) -> list[Mapping[str, Any]]:
        return [e for e in self.edges if e["kind"] == kind]

    def related(self, eid: str, kind: str) -> list[str]:
        out = []
        for e in self.edges:
            if e["kind"] == kind:
                if e["a"] == eid:
                    out.append(e["b"])
                elif e["b"] == eid and kind in ("aligns_with", "overlaps"):
                    out.append(e["a"])
        return out

    def to_dict(self) -> dict[str, Any]:
        def node(n):
            d = dict(n)
            d["box"] = [round(float(v), 4) for v in d["box"]]
            return d

        return {
            "version": self.version,
            "nodes": {k: node(v) for k, v in self.nodes.items()},
            "edges": [dict(e) for e in self.edges],
            "group_policies": [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in self.group_policies],
            "confidence": round(float(self.confidence), 4),
            "provenance": dict(self.provenance),
        }


def _norm_box(b: Box, sw: float, sh: float) -> tuple[float, float, float, float]:
    c = b.clip(sw, sh)
    vals = (c.x / sw, c.y / sh, c.w / sw, c.h / sh)
    return tuple(min(1.0, max(0.0, v)) if math.isfinite(v) else 0.0 for v in vals)  # type: ignore[return-value]


def rotated_footprint(el: dict[str, Any]) -> Box:
    """Axis-aligned bounds of the rotated element only (not unioned with the
    unrotated box, unlike canonical.rotated_extent)."""
    b = element_box(el)
    rot = float(el.get("rotation") or 0.0)
    if abs(rot) < 1e-6:
        return b
    t = math.radians(rot)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    rw, rh = b.w * c + b.h * s, b.w * s + b.h * c
    return Box(b.cx - rw / 2, b.cy - rh / 2, rw, rh)


def _visible(el: dict[str, Any]) -> Box:
    if abs(float(el.get("rotation") or 0)) > 0.5:
        return rotated_footprint(el)
    return glyph_box(el) if el.get("type") == "text" else visual_box(el) if el.get("type") == "image" else element_box(el)


def element_relations(source: dict[str, Any], background_ids: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Role-independent geometric relations between original elements.

    `background_ids` are never containers: a full-canvas background is not a
    group for everything painted on it."""
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    els = list(source["elements"])
    z = {e["id"]: (e.get("z_index", 0), i) for i, e in enumerate(els)}
    bg = set(background_ids)
    vis = {e["id"]: _visible(e) for e in els}
    edges: list[dict[str, Any]] = []
    tol_x, tol_y = 0.012 * sw, 0.012 * sh
    for i, a in enumerate(els):
        va = vis[a["id"]]
        for b in els[i + 1:]:
            vb = vis[b["id"]]
            if va.area <= 0 or vb.area <= 0:
                continue
            lo, hi = (a, b) if z[a["id"]] < z[b["id"]] else (b, a)
            vlo, vhi = vis[lo["id"]], vis[hi["id"]]
            inter = va.intersection_area(vb)
            if inter > 0:
                contained = False
                if lo["id"] not in bg and lo["type"] in ("shape", "image") and vhi.area > 0:
                    inside = vlo.intersection_area(vhi) / vhi.area
                    if inside >= 0.9:
                        cov = alpha_coverage(lo, vhi)
                        if cov >= 0.9:
                            contained = True
                            edges.append({"a": lo["id"], "b": hi["id"], "kind": "contains", "weight": round(min(inside, cov), 3),
                                          "provenance": "geometry:inside>=0.9+alpha>=0.9"})
                if not contained:
                    edges.append({"a": a["id"], "b": b["id"], "kind": "overlaps",
                                  "weight": round(inter / max(1e-6, min(va.area, vb.area)), 3), "provenance": "geometry:visible-intersection"})
            for axis, fa, fb, tol in (
                ("left", va.x, vb.x, tol_x), ("right", va.x1, vb.x1, tol_x), ("cx", va.cx, vb.cx, tol_x),
                ("top", va.y, vb.y, tol_y), ("bottom", va.y1, vb.y1, tol_y), ("cy", va.cy, vb.cy, tol_y),
            ):
                d = abs(fa - fb)
                if d <= tol:
                    edges.append({"a": a["id"], "b": b["id"], "kind": "aligns_with", "axis": axis,
                                  "weight": round(1 - d / max(1e-6, tol), 3), "provenance": "geometry:visible-edges<=1.2%"})
            if len(edges) >= MAX_EDGES:
                return edges
    return edges


def _photo_hosts(source: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for text in source["elements"]:
        if text["type"] != "text" or not str(text.get("content", "")).strip():
            continue
        box = rotated_extent(text)
        best = None
        for e in source["elements"]:
            if e["type"] != "image" or e.get("z_index", 0) >= text.get("z_index", 0):
                continue
            st = asset_stats(e.get("asset_id"))
            if st.kind != "photo":
                continue
            if element_box(e).intersection_area(box) < 0.95 * max(1e-6, box.area):
                continue
            cov = alpha_coverage(e, box)
            if cov < 0.9:
                continue
            if best is None or e.get("z_index", 0) > best[0].get("z_index", 0):
                best = (e, cov)
        if best:
            out.append({"a": best[0]["id"], "b": text["id"], "kind": "supports_on_photo", "weight": round(best[1], 3),
                        "provenance": "pixels:photo-class+alpha>=0.9 under text"})
    return out


def build_graph(source: dict[str, Any], analysis: Any) -> AnalysisGraph:
    """Build (or return the cached) graph for `analysis` of `source`."""
    cached = getattr(analysis, "_graph_cache", None)
    if isinstance(cached, AnalysisGraph):
        return cached
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    els = list(source["elements"])
    zs = sorted({e.get("z_index", 0) for e in els})
    roles = dict(getattr(analysis, "roles", {}) or {})
    sources = dict(getattr(analysis, "role_sources", {}) or {})
    conf = dict(getattr(analysis, "role_confidence", {}) or {})
    owner: dict[str, str] = {}
    blocks = analysis.all_blocks() if hasattr(analysis, "all_blocks") else []
    for b in blocks:
        for i in b.ids:
            owner[i] = b.id
    nodes: dict[str, dict[str, Any]] = {}
    for e in els:
        eid = e["id"]
        role = roles.get(eid, "unknown")
        st = asset_stats(e.get("asset_id")) if e["type"] == "image" else None
        if e["type"] == "image" and st is not None and not st.exists and role in ("accent", "decoration", "image"):
            role = "unknown"  # a missing asset is not evidence of decoration
        c = conf.get(eid, _PRIOR_CONFIDENCE.get(role, 0.4))
        nodes[eid] = {
            "type": e["type"], "role": role, "role_confidence": round(float(min(1.0, max(0.0, c))), 4),
            "role_source": sources.get(eid, "heuristic"),
            "box": _norm_box(_visible(e) if e["type"] != "text" else rotated_extent(e), sw, sh),
            "z": zs.index(e.get("z_index", 0)) / max(1, len(zs) - 1),
            "group": e.get("group_id") or None,
            "block": owner.get(eid),
            "bleeds": bool(element_box(e).x < -1 or element_box(e).y < -1 or element_box(e).x1 > sw + 1 or element_box(e).y1 > sh + 1),
            **({"pixels": st.kind} if st is not None else {}),
        }
    bg = [e["id"] for e in getattr(analysis, "background", [])]
    edges = list(getattr(analysis, "_relations", None) or element_relations(source, bg))
    # Explicit and inferred labels (buttons, badges, label-on-shape groups).
    for b in blocks:
        shape = getattr(b, "shape", None)
        text = getattr(b, "text", None)
        if shape is not None and text is not None:
            edges.append({"a": shape["id"], "b": text["id"], "kind": "label_for", "weight": 1.0,
                          "provenance": "explicit_group" if shape.get("group_id") else "inferred:single-label-on-shape"})
    # Reading order between top-level content blocks.
    content = list(getattr(analysis, "blocks", []))
    for a, b in zip(content, content[1:]):
        edges.append({"a": a.ids[0], "b": b.ids[0], "kind": "reads_before", "weight": 1.0,
                      "provenance": "reading-order:bands(6%)+left-to-right", "blocks": [a.id, b.id]})
    edges.extend(_photo_hosts(source))
    policies = tuple(getattr(analysis, "policies", None) or ())
    if not policies:
        try:
            from .group_policies import policies_from_blocks

            policies = tuple(policies_from_blocks(source, analysis))
        except Exception:  # optional sidecar: never break analysis
            policies = ()
    vals = [n["role_confidence"] for n in nodes.values()]
    graph = AnalysisGraph(
        version=GRAPH_VERSION,
        nodes=nodes,
        edges=tuple(edges[:MAX_EDGES]),
        group_policies=policies,
        confidence=round(sum(vals) / len(vals), 4) if vals else 0.0,
        provenance={
            "builder": "engine.relations.build_graph",
            "group_mode": getattr(analysis, "group_mode", "legacy"),
            "role_status": getattr(analysis, "role_status", {"status": "heuristic"}),
            "features": "geometry, typography metrics, paint order, explicit group_id, asset alpha/colour statistics; never ids or file names",
            "edges_truncated": len(edges) > MAX_EDGES,
        },
    )
    try:
        object.__setattr__(analysis, "_graph_cache", graph)
    except Exception:
        pass
    return graph
