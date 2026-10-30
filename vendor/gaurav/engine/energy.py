"""One energy over a laid-out design: E(layout) = hard penalties
+ preservation terms (the source as prior) + aesthetic terms.

Lower is better. The optimiser (engine/optimize.py) walks a candidate down
this energy; the term vector (`Energy.terms`) is what eval/calibrate.py fits
the weights on (`--energy`). Terms are costs in [0, 1]:

  hard.*          the scorer's hard-fail checks (engine/score.py) as large
                  costs: a content unit off canvas (beyond the source's
                  bleed), measured text overflow, image aspect drift,
                  headline / CTA below the readable floor, an unwanted
                  collision (source-relative, engine/rectify.py), a
                  headline / CTA whose contrast dropped under the drop rule,
                  a text straddling an edge it did not straddle in the source
  pres.align      alignment relations of the source lost (per pair, per axis)
  pres.prox       proximity groups of the source broken
  pres.order      reading-order pairs flipped
  pres.hier       size-hierarchy inversions (score._hierarchy)
  pres.anchor     anchor drift (score._anchor_keep)
  pres.contain    containment / panel carry lost
  pres.symmetry   axis symmetry of the source lost
  pres.scale      non-uniform scaling across units (score._scale_consistency)
  pres.drift      displacement and resize of each unit against the
                  proportional copy of the source (the paper's "previous
                  layout as prior"); it counts in full for a target of the
                  source's shape and fades to nothing at the judge's
                  ADAPT_RATIO (1.75), beyond which the proportional copy is
                  no longer a legitimate layout and a re-arrangement is the point
                  ... every pres.* term is weighted by the importance of the
                  units involved
  aes.align       few distinct alignment lines, every unit aligned to another
  aes.near_miss   edges that almost align (NEAR_MISS_MULT x the tolerance)
  aes.balance     Ngo balance about the canvas centre
  aes.spacing     uneven white space between groups in stacks / rows (gaps
                  inside a proximity group are the group's own business)
  aes.cohesion    an empty stretch between neighbours wider than needed
  aes.fill        content hull and ink density
  aes.margins     breathing room at the edges, text and small pieces only
                  (full-bleed photos and panels are the designer's call, a
                  legal line on the bottom edge a convention)
  aes.hier        hierarchy ratios: the headline clearly larger than the next
                  text, the CTA readable against the body
  aes.read        font sizes vs what the canvas affords (score read.size)
  aes.lines       headline / CTA / subhead line counts (score read.lines)
  aes.contrast    estimated contrast + busyness under the texts
  aes.size        primary units up to AMBITION x proportional (score layout.size)
  aes.presence    hero / logo keep their visual weight (score read.presence)

Weights live in WEIGHTS below (one place); engine/energy_weights.json, when
present and complete, overrides them (eval/calibrate.py --energy writes it;
CHAMELEON_ENERGY_WEIGHTS=0 ignores the file, =<path> reads another).

`Layout` is the mutable unit-level state the optimiser edits (groups move
and scale as one piece, images keep their aspect, fonts follow the scale
and are re-fitted); `evaluate(layout)` returns the Energy for its current
geometry. `energy_of(analysis, source, design, target)` is the one-shot
form for a finished design.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine import contrast as contrast_est
from engine.analyze import (
    Analysis,
    BACKGROUND,
    BADGE,
    BODY,
    CTA,
    FINE_PRINT,
    HEADLINE,
    HERO,
    LOGO,
    SUBHEAD,
    TEXT,
)
from engine.floorplan import Metrics, ratio_family
from engine.geometry import Box, clamp
from engine.rectify import (
    OVERLAP_TOL,
    Live,
    _Rectifier,
    _lives,
    _pair_key,
    is_straddle,
    parts_overlap,
    parts_overlap_fraction,
    rotated_extent,
    visible_box,
)
from engine import relations as R
from engine import score as S
from engine.relayout import ratio_change
from engine.score import (
    ASPECT_TOL,
    ADAPT_RATIO,
    BLEED_TOL,
    CONTRAST_DROP,
    CONTRAST_DROP_REL,
    CONTRAST_FLOOR,
    CONTRAST_GOOD,
    _alignment,
    _anchor_keep,
    _balance,
    _chains,
    _cohesion,
    _fill,
    _hierarchy,
    _margins,
    _order,
    _presence,
    _primary_size,
    _scale_consistency,
    _spacing,
    below_floor,
    source_contrast,
)
from engine.textfit import TextMeasurer, apply_font_size, default_measurer

HARD = 1000.0

WEIGHTS: dict[str, float] = {
    # preservation: the source is the prior (46)
    "pres.align": 8.0,
    "pres.prox": 7.0,
    "pres.order": 10.0,
    "pres.hier": 6.0,
    "pres.anchor": 5.0,
    "pres.contain": 4.0,
    "pres.symmetry": 3.0,
    "pres.scale": 3.0,
    "pres.drift": 8.0,
    # aesthetics (54)
    "aes.align": 6.0,
    "aes.near_miss": 3.0,
    "aes.balance": 5.0,
    "aes.spacing": 4.0,
    "aes.cohesion": 3.0,
    "aes.fill": 5.0,
    "aes.margins": 3.0,
    "aes.hier": 3.0,
    "aes.read": 7.0,
    "aes.lines": 3.0,
    "aes.contrast": 5.0,
    "aes.size": 4.0,
    "aes.presence": 3.0,
}
HAND_WEIGHTS = dict(WEIGHTS)
WEIGHTS_FILE = Path(__file__).with_name("energy_weights.json")

DRIFT_POS_FULL = 0.25        # pres.drift: a unit a quarter of the canvas from its proportional place costs 1 ...
DRIFT_SIZE_FULL = math.log(2.0)   # ... as does a unit twice or half its proportional linear size
DRIFT_POS_SHARE = 0.7        # position vs size inside the drift of one unit
HIER_HEAD_RATIO = 1.4        # aes.hier: the headline wants at least this x the next largest text ...
HIER_SLOPE = 0.4             # ... full cost when it is this much below (ratio 1.0)
TEXT_ROLES = (HEADLINE, SUBHEAD, BODY, CTA, BADGE, FINE_PRINT, TEXT)


def load_weights(path: str | os.PathLike | None = None) -> dict[str, float] | None:
    env = os.environ.get("CHAMELEON_ENERGY_WEIGHTS", "")
    if path is None:
        if env.lower() in ("0", "false", "no", "off"):
            return None
        path = env or WEIGHTS_FILE
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    raw = data.get("weights", data) if isinstance(data, dict) else None
    if not isinstance(raw, dict) or set(raw) != set(HAND_WEIGHTS):
        return None
    try:
        w = {k: float(raw[k]) for k in HAND_WEIGHTS}
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) or v < 0.0 for v in w.values()) or sum(w.values()) <= 0.0:
        return None
    return w


_fitted = load_weights()
if _fitted:
    WEIGHTS.update(_fitted)


@dataclass
class Energy:
    total: float
    hard: list[str] = field(default_factory=list)         # structured "kind:unit[:unit]" keys
    terms: dict[str, float] = field(default_factory=dict)  # every soft term, cost in [0, 1]
    notes: dict[str, Any] = field(default_factory=dict)
    near: list[str] = field(default_factory=list)         # not defects for the judge, but a move may not create them:
                                                           # unrelated pairs closer than the rectifier's gap, text frames past the canvas

    @property
    def soft(self) -> float:
        return self.total - HARD * len(self.hard)

    def hard_set(self) -> frozenset[str]:
        return frozenset(self.hard)

    def as_dict(self) -> dict[str, Any]:
        return {"total": round(self.total, 3), "soft": round(self.soft, 3), "hard": list(self.hard),
                "terms": {k: round(v, 4) for k, v in self.terms.items()}}


def weighted(terms: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = weights or WEIGHTS
    return sum(w.get(k, 0.0) * clamp(v, 0.0, 1.0) for k, v in terms.items())


# ---------------------------------------------------------------------------
# mutable state


class Layout:
    """Unit-level geometry of a materialized design that the optimiser edits.

    Wraps engine/rectify.py's `_Rectifier` for the geometry (visible boxes,
    source-relative collision rule, moves and uniform scales with fonts
    re-fitted) and keeps the element dicts in sync, so every scorer helper
    that reads elements sees the current state."""

    def __init__(self, analysis: Analysis, source: dict[str, Any], elements: list[dict[str, Any]],
                 target: Box, measurer: TextMeasurer | None = None):
        self.analysis = analysis
        self.source = source
        self.target = target
        self.m = measurer or default_measurer()
        self.elements = copy.deepcopy(elements)
        self.by_id = {el["id"]: el for el in self.elements}
        self.src_by_id = {el["id"]: el for el in source["elements"]}
        self.lives = _lives(analysis, self.by_id)
        self.r = _Rectifier(analysis, self.by_id, self.lives, target, self.m)
        self.live_of = {l.unit.key: l for l in self.lives}
        self.met = Metrics.for_canvas(analysis.canvas, target, ratio_family(target))
        self.area_scale = math.sqrt(target.area / analysis.canvas.area)
        self.src_rel = R.source_relations(analysis, self.m)
        self.src_contrast = source_contrast(analysis, source, self.m)
        self.src_straddle = getattr(analysis, "source_straddle", None) or set()
        self._contrast_cache: dict[tuple, tuple[float, float, bool]] = {}
        self._layouts: dict[tuple, Any] = {}
        self.evals = 0

    # -- text measurement (cached by element, font and width) -------------------------

    def text_layout(self, el: dict[str, Any], font: float, width: float):
        key = (el["id"], round(float(font), 3), round(float(width), 2))
        lay = self._layouts.get(key)
        if lay is None:
            lay = self.m.layout(el, font, width)
            if len(self._layouts) > 8192:
                self._layouts.clear()
            self._layouts[key] = lay
        return lay

    def text_fits(self, el: dict[str, Any], font: float, width: float, height: float, slack: float = 0.5) -> bool:
        lay = self.text_layout(el, font, width)
        if el.get("max_lines") and lay.n_lines > int(el["max_lines"]):
            return False
        return lay.max_width <= width + slack and lay.height <= height + slack

    # -- geometry ---------------------------------------------------------------

    def vis(self, l: Live) -> Box:
        return self.r.vis(l)

    def parts(self, l: Live) -> list[Box]:
        return self.r.parts(l)

    def frame(self, l: Live) -> Box:
        return l.frame(self.by_id)

    def _sync(self, l: Live) -> None:
        for eid, b in l.boxes.items():
            el = self.by_id[eid]
            el["x"], el["y"], el["width"], el["height"] = b.x, b.y, b.w, b.h
            if eid in l.fonts:
                apply_font_size(el, l.fonts[eid])

    def move(self, l: Live, dx: float, dy: float) -> None:
        self.r.move(l, dx, dy)
        self._sync(l)

    def scale(self, l: Live, f: float, ax: float | None = None, ay: float | None = None) -> None:
        fb = l.box
        self.r.scale(l, f, fb.cx if ax is None else ax, fb.cy if ay is None else ay)
        self._sync(l)

    def set_text_box(self, l: Live, eid: str, width: float, height: float) -> None:
        """Re-wrap a single text: new box width / height about the same
        top-left (alignment decides the visible shift), font kept."""
        b = l.boxes[eid]
        l.boxes[eid] = Box(b.x, b.y, max(1.0, width), max(1.0, height))
        self.r.invalidate(l)
        self._sync(l)

    def snapshot(self, l: Live) -> tuple[dict[str, Box], dict[str, float]]:
        return dict(l.boxes), dict(l.fonts)

    def restore(self, l: Live, snap: tuple[dict[str, Box], dict[str, float]]) -> None:
        l.boxes, l.fonts = dict(snap[0]), dict(snap[1])
        self.r.invalidate(l)
        self._sync(l)

    def design(self) -> dict[str, Any]:
        return {"canvas": {"width": int(self.target.w), "height": int(self.target.h)},
                "elements": copy.deepcopy(self.elements)}

    # -- evaluation -------------------------------------------------------------------

    def evaluate(self, weights: dict[str, float] | None = None) -> Energy:
        self.evals += 1
        lives = self.lives
        by_id = self.by_id
        target = self.target
        an = self.analysis
        vis = {l.unit.key: self.vis(l) for l in lives}
        hard: list[str] = []
        terms: dict[str, float] = {}
        notes: dict[str, Any] = {}

        # ---- hard ------------------------------------------------------------
        near: list[str] = []
        for l in lives:
            fb = self.frame(l)
            if fb.x < -0.5 or fb.y < -0.5 or fb.right > target.w + 0.5 or fb.bottom > target.h + 0.5:
                off_now = 1.0 - target.contains_fraction(fb)
                off_src = 1.0 - an.canvas.contains_fraction(l.unit.box)
                if off_now > self._bleed_allowance(l, off_src):
                    hard.append(f"offcanvas:{l.unit.key}")
                elif any(by_id[e]["type"] == "text" for e in l.boxes):
                    # The judge excuses a text box that bled in the source too,
                    # but the engine never emits one when it can help it
                    # (resize._finalize clamps unrotated text; a rotated box
                    # that grows past the canvas cannot be clamped): a move
                    # must not create such a bleed (see `new_hard`).
                    near.append(f"bleed:{l.unit.key}")
            for eid, b in l.boxes.items():
                el = by_id[eid]
                if el["type"] == "text":
                    font = l.fonts.get(eid, float(el["font_size"]))
                    if self._overflows(el, font, b):
                        hard.append(f"overflow:{eid}")
                    role = an.roles.get(eid)
                    if role in (HEADLINE, CTA):
                        src_font = float(self.src_by_id[eid]["font_size"]) if eid in self.src_by_id else font
                        if below_floor(font, role, src_font, self.met, self.area_scale):
                            hard.append(f"floor:{eid}")
                elif el["type"] == "image" and el.get("fit", "contain") != "cover":
                    s = self.src_by_id.get(eid)
                    if s and float(s["width"]) > 0 and float(s["height"]) > 0 and b.h > 0:
                        drift = abs((b.w / b.h) / (float(s["width"]) / float(s["height"])) - 1.0)
                        if drift > ASPECT_TOL:
                            hard.append(f"aspect:{eid}")
        for i, a in enumerate(lives):
            pa = self.parts(a)
            for b in lives[i + 1:]:
                pb = self.parts(b)
                if self._conflict(a, b, pa, pb):
                    hard.append(f"collide:{a.unit.key}:{b.unit.key}")
                elif self._near(a, b, pa, pb):
                    near.append(f"{a.unit.key}:{b.unit.key}")

        # ---- contrast (estimate, cached per text and layer configuration) ----
        ratios, busy, straddles = self._contrast()
        c_num = c_den = 0.0
        for eid, ratio in ratios.items():
            role = an.roles.get(eid, TEXT)
            imp = max(an.unit(an.element_unit[eid]).importance, 0.05) if eid in an.element_unit else 0.3
            bz = clamp((busy.get(eid, 0.0) - 20.0) / 60.0, 0.0, 1.0)
            c_num += imp * clamp((ratio - CONTRAST_FLOOR) / (CONTRAST_GOOD - CONTRAST_FLOOR), 0.0, 1.0) * (1.0 - 0.5 * bz)
            c_den += imp
            if role in (HEADLINE, CTA):
                src_ratio = self.src_contrast.get(eid)
                if ratio < CONTRAST_DROP and (src_ratio is None or ratio < CONTRAST_DROP_REL * src_ratio):
                    hard.append(f"contrast:{eid}")
        for eid in straddles:
            if an.roles.get(eid) in (HEADLINE, CTA, SUBHEAD, BODY, TEXT):
                hard.append(f"straddle:{eid}")
        terms["aes.contrast"] = 1.0 - (c_num / c_den if c_den else 1.0)

        # ---- preservation ------------------------------------------------------
        out_rel = R.relations(an, self.elements, target, self.m, boxes=vis)
        src_rel = self.src_rel
        imp = src_rel.importance
        pair_w = lambda a, b: min(imp.get(a, 0.3), imp.get(b, 0.3))  # noqa: E731
        align_src = sum(pair_w(a, b) * sum(1 for ax in ("x", "y") if src_rel.shares_line(a, b, ax))
                        for (a, b) in src_rel.aligned_pairs)
        lost = R.lost_alignments(src_rel, out_rel)
        terms["pres.align"] = (sum(pair_w(a, b) for a, b, _ in lost) / align_src) if align_src else 0.0
        prox_src = sum(pair_w(a, b) for (a, b) in src_rel.proximity)
        broken = R.broken_proximity(src_rel, out_rel)
        terms["pres.prox"] = (sum(pair_w(a, b) * c for a, b, c in broken) / prox_src) if prox_src else 0.0
        terms["pres.order"], flipped = R.order_discordance(src_rel, out_rel)
        terms["pres.hier"] = 1.0 - _hierarchy(self.elements, self.src_by_id, an.roles)
        terms["pres.anchor"] = 1.0 - _anchor_keep(lives, target)
        cont_src = sum(imp.get(a, 0.3) for a in src_rel.containment)
        lost_c = R.lost_containment(src_rel, out_rel)
        terms["pres.contain"] = (sum(imp.get(a, 0.3) for a, _ in lost_c) / cont_src) if cont_src else 0.0
        sym_src = sum(imp.get(k, 0.3) * (int(sx) + int(sy)) for k, (sx, sy) in src_rel.symmetry.items())
        lost_s = R.lost_symmetry(src_rel, out_rel)
        terms["pres.symmetry"] = (sum(imp.get(k, 0.3) for k, _ in lost_s) / sym_src) if sym_src else 0.0
        terms["pres.scale"] = 1.0 - _scale_consistency(lives)
        terms["pres.drift"] = self._drift(vis)
        notes["lost_alignments"] = lost
        notes["broken_proximity"] = broken
        notes["flipped_order"] = flipped

        # ---- aesthetics ----------------------------------------------------------
        boxes = [vis[l.unit.key] for l in lives]
        text_boxes = [vis[l.unit.key] for l in lives if self._margin_judged(l)]
        no_legal = [vis[l.unit.key] for l in lives if l.unit.role != FINE_PRINT]
        terms["aes.align"] = 1.0 - _alignment(boxes, target)
        n_pairs = max(1, len(boxes) * (len(boxes) - 1) // 2)
        terms["aes.near_miss"] = clamp(len(out_rel.near_misses) / n_pairs, 0.0, 1.0)
        terms["aes.balance"] = 1.0 - _balance(lives, target, vis)
        terms["aes.spacing"] = 1.0 - group_spacing(no_legal, target)
        terms["aes.cohesion"] = 1.0 - _cohesion(no_legal, target)
        terms["aes.fill"] = 1.0 - _fill(boxes, target)
        terms["aes.margins"] = 1.0 - _margins([vis[l.unit.key] for l in lives
                                               if self._margin_judged(l) and l.unit.role != FINE_PRINT], target)
        terms["aes.hier"] = self._hierarchy_ratios()
        read, lines = self._readability()
        terms["aes.read"] = 1.0 - read
        terms["aes.lines"] = 1.0 - lines
        terms["aes.size"] = 1.0 - _primary_size(lives, an, by_id, self.src_by_id, target, self.area_scale)
        terms["aes.presence"] = 1.0 - _presence(lives, an.canvas, target)

        hard = sorted(set(hard))
        total = weighted(terms, weights) + HARD * len(hard)
        return Energy(total=total, hard=hard, terms=terms, notes=notes, near=sorted(set(near)))

    # -- the judge's hard rules (engine/score.py, engine/rectify.py), shared ----------
    # The v6 helpers are used when present so the energy and the judge agree
    # rule for rule; the v5 rules are the fallback.

    def _conflict(self, a: Live, b: Live, pa: list[Box], pb: list[Box]) -> bool:
        """The judge's collision rule (no minimum gap: that is `_near`)."""
        rules = getattr(self.r, "rules", None)
        if rules is not None:
            return bool(rules.conflict(a, b, pa, pb))
        if not parts_overlap(pa, pb, tol=1.0):
            return False
        return parts_overlap_fraction(pa, pb) - self.r.src_frac.get(_pair_key(a.unit.key, b.unit.key), 0.0) > OVERLAP_TOL

    def _near(self, a: Live, b: Live, pa: list[Box], pb: list[Box]) -> bool:
        """Two unrelated units closer than the rectifier's minimum gap but
        not colliding. Not a defect for the judge; the optimiser must not
        create one (it would undo what the rectifier just separated)."""
        if self.r.src_frac.get(_pair_key(a.unit.key, b.unit.key), 0.0) > 0.0:
            return False
        return parts_overlap(pa, pb, tol=-self.r.gap) and not parts_overlap(pa, pb, tol=1.0)

    def _overflows(self, el: dict[str, Any], font: float, b: Box) -> bool:
        fn = getattr(S, "text_overflows", None)
        if fn is not None:
            probe = dict(el, x=b.x, y=b.y, width=b.w, height=b.h, font_size=font)
            return bool(fn(probe, self.m))
        return not self.text_fits(el, font, b.w, b.h)

    def _bleed_allowance(self, l: Live, off_src: float) -> float:
        fn = getattr(S, "bleed_allowance", None)
        if fn is not None:
            return float(fn(l.unit, off_src, self.src_by_id))
        return off_src + BLEED_TOL

    def _margin_judged(self, l: Live) -> bool:
        fn = getattr(S, "margin_judged", None)
        if fn is not None:
            return bool(fn(l.unit, self.src_by_id, self.analysis.canvas))
        return any(self.by_id[e]["type"] == "text" for e in l.unit.ids if e in self.by_id)

    # -- pieces -------------------------------------------------------------------

    def _readability(self) -> tuple[float, float]:
        """score_design's read.size / read.lines on the current state."""
        an = self.analysis
        size_num = size_den = lines_num = lines_den = 0.0
        for l in self.lives:
            for eid in l.boxes:
                el = self.by_id[eid]
                if el["type"] != "text":
                    continue
                role = an.roles.get(eid, TEXT)
                imp = max(l.unit.importance, 0.05)
                src_font = float(self.src_by_id[eid]["font_size"]) if eid in self.src_by_id else float(el["font_size"])
                ideal = max(src_font * self.area_scale, self.met.min_font(role))
                size_num += imp * clamp(float(el["font_size"]) / ideal, 0.0, 1.0)
                size_den += imp
                if role in (HEADLINE, CTA, SUBHEAD):
                    n = self.text_layout(el, float(el["font_size"]), float(el["width"])).n_lines
                    allowed, slope = {HEADLINE: (3, 2.0), CTA: (1, 2.0)}.get(role, (3, 5.0))
                    lines_num += imp * clamp(1.0 - max(0, n - allowed) / slope, 0.0, 1.0)
                    lines_den += imp
        return (size_num / size_den if size_den else 1.0), (lines_num / lines_den if lines_den else 1.0)

    def _drift(self, vis: dict[str, Box]) -> float:
        """Importance-weighted drift of the units from the proportional copy
        of the source (centre per axis, linear size share), fading with the
        shape change: full at the source's shape, zero from ADAPT_RATIO on
        (where the judge starts taxing un-re-laid-out anchor layouts)."""
        change = ratio_change(self.analysis.canvas, self.target)
        fade = clamp((ADAPT_RATIO - change) / (ADAPT_RATIO - 1.0), 0.0, 1.0)
        if fade <= 0.0:
            return 0.0
        src = self.src_rel
        num = den = 0.0
        for l in self.lives:
            k = l.unit.key
            if k not in src.boxes or k not in vis:
                continue
            sb, ob = src.boxes[k], vis[k]
            ex, ey = sb.cx / src.canvas.w, sb.cy / src.canvas.h
            pos = math.hypot(ob.cx / self.target.w - ex, ob.cy / self.target.h - ey)
            s_share = max(src.size.get(k, 0.0), 1e-6)
            o_share = max(math.sqrt(max(ob.area, 0.0) / self.target.area), 1e-6)
            size = abs(math.log(o_share / s_share))
            d = DRIFT_POS_SHARE * clamp(pos / DRIFT_POS_FULL, 0.0, 1.0) + (1.0 - DRIFT_POS_SHARE) * clamp(size / DRIFT_SIZE_FULL, 0.0, 1.0)
            w = max(l.unit.importance, 0.05)
            num += w * d
            den += w
        return fade * (num / den if den else 0.0)

    def _hierarchy_ratios(self) -> float:
        """The headline should read as the largest text by a margin
        (GUIDELINES: roughly 1 : 1.6 between levels); cost rises linearly as
        the ratio to the next largest text falls from HIER_HEAD_RATIO to 1."""
        an = self.analysis
        head = None
        others: list[float] = []
        for l in self.lives:
            for eid in l.boxes:
                el = self.by_id[eid]
                if el["type"] != "text" or not str(el.get("content", "")).strip():
                    continue
                f = float(el["font_size"])
                if an.roles.get(eid) == HEADLINE and head is None:
                    head = f
                elif an.roles.get(eid) != HEADLINE:
                    others.append(f)
        if head is None or not others:
            return 0.0
        ratio = head / max(1.0, max(others))
        return clamp((HIER_HEAD_RATIO - ratio) / HIER_SLOPE, 0.0, 1.0)

    def _contrast(self) -> tuple[dict[str, float], dict[str, float], list[str]]:
        """Per-text contrast ratio, busyness and the new straddles from the
        JSON estimate, cached by (ink box, layers under it)."""
        if not contrast_est.available():
            return {}, {}, []
        # The cache key of a text is its ink box plus the non-text elements
        # that lie under it: moving the hero re-estimates only the texts it
        # touched, moving a text only that text.
        layers = []
        for el in self.elements:
            if el["type"] != "text":
                ext = rotated_extent(Box.of(el), float(el.get("rotation", 0.0) or 0.0))
                layers.append((el["id"], round(ext.x, 1), round(ext.y, 1), round(ext.w, 1), round(ext.h, 1), ext))
        ink: dict[str, Box] = {}
        keys: dict[str, tuple] = {}
        for l in self.lives:
            for eid, b in l.boxes.items():
                el = self.by_id[eid]
                if el["type"] != "text" or not str(el.get("content", "")).strip():
                    continue
                box = visible_box(el, b, l.fonts.get(eid, float(el["font_size"])), self.m)
                ink[eid] = box
                probe = Box(box.x - 2.0, box.y - 2.0, box.w + 4.0, box.h + 4.0)
                under = tuple(lay[:5] for lay in layers if lay[5].overlaps(probe))
                keys[eid] = (eid, round(box.x, 1), round(box.y, 1), round(box.w, 1), round(box.h, 1),
                             round(l.fonts.get(eid, float(el["font_size"])), 1), under)
        todo = {eid for eid, k in keys.items() if k not in self._contrast_cache}
        if todo:
            design = {"canvas": {"width": self.target.w, "height": self.target.h}, "elements": self.elements}
            est = contrast_est.estimate(design, {e: ink[e] for e in todo}, ids=todo)
            for eid in todo:
                tc = est.get(eid)
                if tc is None:
                    self._contrast_cache[keys[eid]] = (float("nan"), 0.0, False)
                else:
                    self._contrast_cache[keys[eid]] = (tc.ratio, tc.busy, is_straddle(tc.slices or []))
            if len(self._contrast_cache) > 4096:
                self._contrast_cache = {k: self._contrast_cache[k] for k in keys.values() if k in self._contrast_cache}
        ratios, busy, straddles = {}, {}, []
        for eid, k in keys.items():
            ratio, bz, st = self._contrast_cache[k]
            if ratio != ratio:   # nan: no estimate for this text
                continue
            ratios[eid], busy[eid] = ratio, bz
            if st and eid not in self.src_straddle:
                straddles.append(eid)
        return ratios, busy, straddles


# ---------------------------------------------------------------------------
# aesthetic terms of our own (the rest are engine/score.py's)


def group_spacing(boxes: list[Box], target: Box) -> float:
    """Even white space *between groups*. The scorer's `_spacing` wants every
    gap of a stack or row equal; designers deliberately keep related pieces
    tight and unrelated ones apart (the designer references lose on that
    term in 79 % of the cases), so here a gap no wider than the proximity
    threshold (engine/relations.py) is a group's internal gap and is left
    out; the remaining gaps of a chain should be even. 1.0 when fewer than
    two inter-group gaps exist."""
    scores = []
    for axis in ("y", "x"):
        span = target.h if axis == "y" else target.w
        for chain in _chains(boxes, axis):
            if len(chain) < 3:
                continue
            gaps = []
            for a, b in zip(chain, chain[1:]):
                if axis == "y":
                    gap, ext = b.y - a.bottom, min(a.h, b.h)
                else:
                    gap, ext = b.x - a.right, min(a.w, b.w)
                if gap > max(R.PROX_SIZE * ext, R.PROX_CANVAS * span):
                    gaps.append(gap)
            if len(gaps) < 2:
                continue
            mean = sum(gaps) / len(gaps)
            if mean <= 0:
                continue
            sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
            scores.append(clamp(1.0 - sd / mean, 0.0, 1.0))
    return sum(scores) / len(scores) if scores else 1.0


def evaluate(layout: Layout, weights: dict[str, float] | None = None) -> Energy:
    return layout.evaluate(weights)


def energy_of(analysis: Analysis, source: dict[str, Any], design: dict[str, Any],
              target: Box | None = None, measurer: TextMeasurer | None = None,
              weights: dict[str, float] | None = None) -> Energy:
    """The energy of a finished design (one-shot form)."""
    target = target or Box(0.0, 0.0, float(design["canvas"]["width"]), float(design["canvas"]["height"]))
    return Layout(analysis, source, design["elements"], target, measurer).evaluate(weights)


def new_hard(seed: Energy, now: Energy) -> list[str]:
    """Hard fails `now` has that the seed did not (the optimiser's invariant),
    plus new sub-gap near-touches between unrelated units (`Energy.near`:
    not a defect for the judge, but a move must not create one)."""
    before = seed.hard_set()
    out = [h for h in now.hard if h not in before]
    near_before = set(seed.near)
    out += [f"near:{n}" for n in now.near if n not in near_before]
    return out


__all__ = ["Energy", "Layout", "WEIGHTS", "HAND_WEIGHTS", "HARD", "evaluate", "energy_of", "new_hard",
           "load_weights", "weighted"]
