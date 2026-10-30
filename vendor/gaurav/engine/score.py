"""Judge a materialized candidate the way the official scorer will.

Ordered like the rubric:

  1. hard fails   (each costs HARD_PENALTY, so any candidate without one beats
                   any candidate with one): a content unit off canvas, text
                   that overflows its box (measured), an image whose aspect
                   drifted > 5 %, a headline / CTA below the readable minimum
                   for this canvas, an unwanted collision between content
                   units, unreadable contrast under a headline / CTA
  2. readability  font sizes relative to what the canvas affords, few lines
                   for headline / CTA, contrast of the text colour against the
                   pixels under the text (sampled on a render without text)
                   and how busy those pixels are
  3. layout       Ngo-style balance, alignment, even spacing, white-space /
                   fill (content hull and ink density), breathing room at
                   the edges, and (v4) the size of the primary units
                   (headline, CTA, hero, logo) up to AMBITION x the
                   source-proportional size: among clean layouts the one
                   that uses the canvas wins
  4. faithfulness reading order, which edges things hug, size hierarchy, and
                   how uniformly everything scaled

The result is one comparable number (`Score.total`, higher is better) plus
a breakdown dict for debugging / the playground. Rendering is optional:
without an image the contrast terms are neutral and the breakdown says so.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine import backdrop
from engine import contrast as contrast_est
from engine.analyze import (
    ROW_HEIGHT_TOL,
    ROW_MAX_GAP,
    ROW_Y_OVERLAP,
    Analysis,
    BADGE,
    BODY,
    CTA,
    FINE_PRINT,
    HEADLINE,
    HERO,
    LOGO,
    SUBHEAD,
    TEXT,
    Unit,
    _anchors,
    _rgb,
)
from engine.floorplan import AMBITION, Metrics, ratio_family
from engine.geometry import Box, clamp
from engine.rectify import Live, _lives, unit_visible_box, unit_visible_parts, unwanted_overlaps, visible_box
from engine.rectify import is_straddle as _is_straddle_slices
from engine.relayout import ratio_change
from engine.render import image_scale, render_design
from engine.textfit import TextMeasurer, default_measurer

HARD_PENALTY = 1000.0
ADAPT_RATIO = 1.75           # ratio change from which a non-re-laid-out (anchor) candidate ...
ADAPT_PENALTY = 18.0         # ... pays this (fernando-ace's figure); it still wins when it is the only clean one
DISRUPT_RATIO = 1.25         # below this ratio change a re-laid-out (floor plan) candidate pays up to ...
DISRUPT_PENALTY = 4.0        # ... this (linearly, 0 at DISRUPT_RATIO): a finished design is not re-arranged for nothing
FLOOR_REL = 0.98             # a font is below the floor only if it also shrank below this x the source font x canvas scale
BLEED_TOL = 0.01             # a unit may be off canvas by this fraction more than it was in the source
BLEED_SRC_MIN = 0.01         # v6: a picture / shape unit that bled at least this much in the source ...
BLEED_GROW = 1.5             # ... may bleed up to this x as much (a photo cropped by the edge stays cropped) ...
BLEED_CAP = 0.25             # ... but never more than this share of itself (nothing important leaves the canvas)
BIG_UNIT_FRAC = 0.15         # v6: a picture / shape unit covering this share of the canvas is a block, not a margin matter
OVERFLOW_SLACK = 0.5         # px of painted ink past the box that still counts as fitting (the designer DSL's tolerance)
HIER_RATIO = 0.82            # higher-ranked text keeps >= min(this, source ratio) of the lower one's size
# v8: row integrity (faith.rows). A sibling row the analyzer found in the
# source (chips side by side: badges, thumbnails, icons) stays one line of
# parts: consecutive parts overlap on the cross axis by ROW_KEEP_OVERLAP of
# the smaller one (a row may become a column on a strip), no gap wider than
# ROW_KEEP_GAP x the larger part (or the source's own ratio x ROW_KEEP_GROW),
# cross-axis extents within ROW_KEEP_SPREAD (or the source's spread), and gaps
# as even as the source's plus ROW_KEEP_CV. A row that wraps into a compact
# cluster (2 + 1 on a banner) is kept too: every part has a neighbour within
# the same gap allowance (nearest-neighbour gap relative to the parts'
# thickness). Two chips pinned to opposite corners fail both tests.
ROW_KEEP_OVERLAP = 0.5
ROW_KEEP_GAP = ROW_MAX_GAP          # 1.5 x the larger part (the analyzer's row criterion)
ROW_KEEP_GROW = 1.25                # a gap may grow this much past the source's gap-to-height ratio
ROW_KEEP_SPREAD = ROW_HEIGHT_TOL    # 0.25
ROW_KEEP_CV = 0.15
# v8: structure prior. When a preserve-structure candidate *survives* a big
# shape change (ratio change >= ADAPT_RATIO) - no hard fail, and its
# readability block holds: read.size, read.lines and read.presence (text at
# its proportional size, headline / CTA line counts, hero and logo keeping
# their weight) all at least STRUCTURE_READ_MIN - the
# template (floor plan) candidates, and the structure variants that did not
# survive themselves, pay this: the source's own composition, re-proportioned,
# is the better default than a template, and the designer references say so
# (docs/EVAL_V8.md). A composition that only fits by halving its copy, wrapping
# the headline on five lines or shrinking the product to a 12 px sticker has
# not survived: it earns nothing and,
# when no variant survived, the templates compete unpenalised. Applied by
# engine/resize.py like the adaptation penalty (it needs the whole set).
STRUCTURE_PRIOR = 3.0
STRUCTURE_READ_MIN = 0.8
SURVIVAL_TERMS = ("read.size", "read.lines", "read.presence")

WEIGHTS = {
    # readability (32)
    "read.size": 11.0,
    "read.lines": 6.0,
    "read.contrast": 8.0,
    "read.presence": 6.0,     # hero / logo keep their visual weight
    # layout quality (43)
    "layout.balance": 6.0,
    "layout.alignment": 7.0,
    "layout.spacing": 5.0,
    "layout.fill": 6.0,
    "layout.margins": 4.0,
    "layout.size": 8.0,       # v4: primary units up to AMBITION x the proportional size
    "layout.cohesion": 4.0,   # v5: no empty stretch between neighbours wider than the content needs
    # faithfulness to the source (29)
    "faith.order": 12.0,
    "faith.anchors": 6.0,
    "faith.hierarchy": 8.0,
    "faith.scale": 3.0,
    "faith.rows": 4.0,        # v8: a sibling row stays a row (or a column), no corner scatter
}
HAND_WEIGHTS = dict(WEIGHTS)                                  # the hand-set defaults above
# v10: `read.backdrop` (engine/backdrop.py) joins the breakdown when
# CHAMELEON_BACKGROUND is on; its weight is taken from the layout block in
# proportion so the sum stays 100 (`active_weights`); the fitted file and
# the hand weights above are untouched, and with the flag off the term is
# absent and the weights are exactly WEIGHTS.
BACKDROP_WEIGHT = 3.0
WEIGHTS_FILE = Path(__file__).with_name("score_weights.json")   # written by eval/calibrate.py


def load_weights(path: str | os.PathLike | None = None) -> dict[str, float] | None:
    """Weights fitted on human labels (eval/calibrate.py, docs/CALIBRATION.md)
    when `path` (default engine/score_weights.json) exists and holds a
    finite, non-negative value for every component; otherwise None and the
    hand weights stay. CHAMELEON_SCORE_WEIGHTS=0 ignores the file,
    CHAMELEON_SCORE_WEIGHTS=<path> reads another one."""
    env = os.environ.get("CHAMELEON_SCORE_WEIGHTS", "")
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


def active_weights() -> dict[str, float]:
    """The weights in force: WEIGHTS, plus `read.backdrop` (v10) when the
    backdrop judge term is on, paid for by the layout block."""
    if not backdrop.enabled("judge"):
        return WEIGHTS
    try:
        bw = float(os.environ.get("CHAMELEON_BACKDROP_WEIGHT", "") or BACKDROP_WEIGHT)   # dev knob for the measurement
    except ValueError:
        bw = BACKDROP_WEIGHT
    w = dict(WEIGHTS)
    layout = [k for k in w if k.startswith("layout.")]
    total = sum(w[k] for k in layout)
    if total > bw:
        f = (total - bw) / total
        for k in layout:
            w[k] *= f
    w["read.backdrop"] = bw
    return w

ASPECT_TOL = 0.05
CONTRAST_GOOD = 4.5          # WCAG AA for normal text
CONTRAST_FLOOR = 1.5         # below: no score
CONTRAST_DROP = 3.0          # a headline / CTA under this ratio ...
CONTRAST_DROP_REL = 0.5      # ... and under this share of its own contrast in the source is a hard fail
CONTRAST_FAIL = 1.3          # kept for reference: the old absolute rule (the drop rule replaces it)
GOOD_FILL = 0.55             # content hull covering this much of the canvas is "filled"
COHESION_GAP_MULT = 3.0      # layout.cohesion: a gap between neighbours up to max(this x the base gap, ...
COHESION_GAP_FRAC = 0.08     # ... this x the canvas span) is free; beyond that the term drops (0 at half the span)
# Straddles (v5): see engine/rectify.py (STRADDLE_MIN / STRADDLE_MAX / STRADDLE_RATIO, `is_straddle`).
GOOD_DENSITY = 0.30          # visible boxes covering this share of the canvas is "used" (v4, fill's second half)
DENSITY_SHARE = 0.4          # weight of the density half inside layout.fill
SIZE_WEIGHTS = {HEADLINE: 1.0, HERO: 1.0, CTA: 0.8, LOGO: 0.5}   # layout.size: who counts, and how much
TEXT_ROLES = (HEADLINE, SUBHEAD, BODY, CTA, BADGE, FINE_PRINT, TEXT)
RANK = {HEADLINE: 4, SUBHEAD: 3, BODY: 2, CTA: 2, TEXT: 2, BADGE: 1, FINE_PRINT: 0}


@dataclass
class Score:
    total: float
    hard: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    penalty: float = 0.0        # soft penalties outside the 0-100 breakdown (adaptation)

    @property
    def soft(self) -> float:
        return self.total + HARD_PENALTY * len(self.hard) + self.penalty

    def as_dict(self) -> dict[str, Any]:
        return {"total": round(self.total, 2), "soft": round(self.soft, 2), "hard": list(self.hard),
                "penalty": round(self.penalty, 2),
                "breakdown": {k: round(v, 3) for k, v in self.breakdown.items()}, "notes": self.notes}


def score_design(analysis: Analysis, source: dict[str, Any], design: dict[str, Any],
                 target: Box | None = None, image=None, render: bool = True,
                 measurer: TextMeasurer | None = None, strategy: str | None = None) -> Score:
    """Score `design` (a resize of `source`, analysed as `analysis`). Pass a
    PIL image rendered without text to reuse a render; with `render=True`
    and no image one is rendered here (skipped silently if unavailable).
    Without pixels the contrast terms come from the no-render estimate
    (engine/contrast.py). `strategy` ("anchor" / "floorplan") enables the
    adaptation penalty for layouts that were not re-laid-out on a big
    shape change."""
    m = measurer or default_measurer()
    target = target or Box(0.0, 0.0, float(design["canvas"]["width"]), float(design["canvas"]["height"]))
    elements = design["elements"]
    by_id = {el["id"]: el for el in elements}
    src_by_id = {el["id"]: el for el in source["elements"]}
    lives = _lives(analysis, by_id)
    vis = {l.unit.key: unit_visible_box(l, by_id, m) for l in lives}
    hard: list[str] = []
    bd: dict[str, float] = {}
    notes: dict[str, Any] = {}

    # ---- 1. hard fails -------------------------------------------------------
    src_ids = [el["id"] for el in source["elements"]]
    if [el["id"] for el in elements] != src_ids:
        hard.append("element ids differ from the source")
    for l in lives:
        fb = l.frame(by_id)    # drawn extent: a rotated box counts rotated
        if fb.x < -0.5 or fb.y < -0.5 or fb.right > target.w + 0.5 or fb.bottom > target.h + 0.5:
            # Bleed the designer had (a photo running off the edge) is excused
            # up to the same share; anything more is a defect.
            off_now = 1.0 - target.contains_fraction(fb)
            off_src = 1.0 - analysis.canvas.contains_fraction(l.unit.box)
            if off_now > bleed_allowance(l.unit, off_src, src_by_id):
                hard.append(f"{l.unit.key} off canvas")
    met = Metrics.for_canvas(analysis.canvas, target, ratio_family(target))
    area_scale = math.sqrt(target.area / analysis.canvas.area)
    for el in elements:
        if el["type"] == "text":
            font = float(el["font_size"])
            if text_overflows(el, m):
                hard.append(f"{el['id']} text overflows its box")
            role = analysis.roles.get(el["id"])
            src_font = float(src_by_id[el["id"]]["font_size"]) if el["id"] in src_by_id else font
            if role in (HEADLINE, CTA) and below_floor(font, role, src_font, met, area_scale):
                hard.append(f"{el['id']} {role} font {font:g} below readable {met.min_font(role):g}")
        elif el["type"] == "image" and el.get("fit", "contain") != "cover":
            s = src_by_id.get(el["id"])
            if s and float(s["width"]) > 0 and float(s["height"]) > 0 and float(el["height"]) > 0:
                drift = abs((float(el["width"]) / float(el["height"])) / (float(s["width"]) / float(s["height"])) - 1.0)
                if drift > ASPECT_TOL:
                    hard.append(f"{el['id']} image aspect drifted {drift:.0%}")
    overlaps = unwanted_overlaps(analysis, elements, m)
    for a, b, area in overlaps:
        hard.append(f"{a} collides with {b} ({area:.0f}px²)")
    from engine import flags
    if flags.enabled("visible"):
        # v12: what the source showed must still be visible in the render
        from engine.visibility import hidden_content
        for eid, cid, frac in hidden_content(analysis, elements, m):
            hard.append(f"{eid} hidden under {cid} ({frac:.0%})")

    # ---- 2. readability ------------------------------------------------------
    size_num = size_den = 0.0
    lines_num = lines_den = 0.0
    for el in elements:
        if el["type"] != "text":
            continue
        role = analysis.roles.get(el["id"], TEXT)
        imp = analysis.unit(analysis.element_unit[el["id"]]).importance if el["id"] in analysis.element_unit else 0.3
        imp = max(imp, 0.05)
        src_font = float(src_by_id[el["id"]]["font_size"]) if el["id"] in src_by_id else float(el["font_size"])
        ideal = max(src_font * area_scale, met.min_font(role))
        size_num += imp * clamp(float(el["font_size"]) / ideal, 0.0, 1.0)
        size_den += imp
        if role in (HEADLINE, CTA, SUBHEAD):
            # A headline on more than three lines, a button label on more than
            # one, reads as a defect however the source wrapped them.
            n = m.layout(el).n_lines
            allowed, slope = {HEADLINE: (3, 2.0), CTA: (1, 2.0)}.get(role, (3, 5.0))
            extra = max(0, n - allowed)
            lines_num += imp * clamp(1.0 - extra / slope, 0.0, 1.0)
            lines_den += imp
    bd["read.size"] = size_num / size_den if size_den else 1.0
    bd["read.lines"] = lines_num / lines_den if lines_den else 1.0
    bd["read.presence"] = _presence(lives, analysis.canvas, target)

    if image is None and render:
        try:
            image = render_design(design, with_text=False)
        except Exception as exc:  # the renderer is a bonus, never a dependency
            image = None
            notes["render_error"] = str(exc)
    est: dict[str, Any] = {}
    try:
        contrast, worst, src_worst, how, est = _contrast_terms(analysis, source, elements, design, image, m)
    except Exception as exc:  # pragma: no cover - odd image / crop
        contrast, worst, src_worst, how = 1.0, {}, {}, "error"
        notes["contrast_error"] = str(exc)
        image = None
    bd["read.contrast"] = contrast
    if backdrop.enabled("judge"):
        # v10: the backdrop under the headline / subhead / CTA looks like
        # the source's (band, glow, texture, hue); pixels when this stage
        # has them, the JSON composite otherwise, like with like.
        try:
            bd["read.backdrop"], notes["backdrop"] = _backdrop_term(analysis, source, elements, design, image, est, m)
        except Exception as exc:  # pragma: no cover
            bd["read.backdrop"], notes["backdrop"] = 1.0, {"error": str(exc)}
    notes["contrast"] = worst
    notes["contrast_source"] = how
    if image is None:
        notes["pixels"] = "no render; contrast from the JSON estimate" if worst else "no render; contrast neutral"
    for eid, ratio in worst.items():
        role = analysis.roles.get(eid)
        if role not in (HEADLINE, CTA):
            continue
        src_ratio = src_worst.get(eid)
        # The drop rule: unreadable now AND clearly worse than the designer's
        # own version (stand-in assets or a loosely built source are not our fault).
        if ratio < CONTRAST_DROP and (src_ratio is None or ratio < CONTRAST_DROP_REL * src_ratio):
            hard.append(f"{eid} unreadable contrast {ratio:.2f} (source {src_ratio if src_ratio is None else round(src_ratio, 2)})")
    # Straddles (v5): a text half over a photo and half on the canvas reads
    # badly whatever its 25th percentile says; the source-relative part
    # excuses a text the designer already laid across an edge.
    straddles = notes["straddles"] = straddling_texts(analysis, source, elements, design, m)
    for eid in straddles:
        role = analysis.roles.get(eid)
        if role in (HEADLINE, CTA, SUBHEAD, BODY, TEXT):
            hard.append(f"{eid} straddles an edge (one part readable, one not)")

    # ---- 3. layout quality ---------------------------------------------------
    # On what the viewer sees: painted text, rotated extents (a text box is
    # taller than its glyphs and often wider than its longest line).
    boxes = [vis[l.unit.key] for l in lives]
    bd["layout.balance"] = _balance(lives, target, vis)
    bd["layout.alignment"] = _alignment(boxes, target)
    # Legal copy sits on the bottom edge by convention; the gap above it is
    # not a rhythm defect (v4: it cost hero-beside-column layouts 3 points).
    # The designer's own rhythm is the baseline (v6): a header / band /
    # footer composition has uneven gaps in the source, and keeping them is
    # not a defect; only unevenness beyond the source's is penalised.
    bd["layout.spacing"] = _spacing([vis[l.unit.key] for l in lives if l.unit.role != FINE_PRINT],
                                    free=source_spacing_cv(analysis, m))
    bd["layout.fill"] = _fill(boxes, target)
    # Breathing room is a matter for text and small pieces (v6): a photo
    # band or a colour block run to the edge is how designers compose.
    bd["layout.margins"] = _margins([vis[l.unit.key] for l in lives if margin_judged(l.unit, src_by_id, analysis.canvas)],
                                    target)
    bd["layout.size"] = _primary_size(lives, analysis, by_id, src_by_id, target, area_scale)
    bd["layout.cohesion"] = _cohesion([vis[l.unit.key] for l in lives if l.unit.role != FINE_PRINT], target)

    # ---- 4. faithfulness -----------------------------------------------------
    bd["faith.order"] = _order(lives, vis)
    bd["faith.anchors"] = _anchor_keep(lives, target)
    bd["faith.hierarchy"] = _hierarchy(elements, src_by_id, analysis.roles)
    bd["faith.scale"] = _scale_consistency(lives)
    bd["faith.rows"] = _row_integrity(lives, analysis, by_id, m)

    # ---- 5. adaptation -----------------------------------------------------
    # A layout that only slid things to their anchors is not a design for a
    # canvas of a very different shape; it still wins when it is the only
    # candidate without a hard defect (the penalty is far below HARD_PENALTY).
    penalty = 0.0
    change = ratio_change(analysis.canvas, target)
    if strategy == "anchor" and change >= ADAPT_RATIO:
        penalty = ADAPT_PENALTY
        notes["adaptation_penalty"] = penalty
    elif strategy in ("floorplan", "structure") and change < DISRUPT_RATIO:
        penalty = DISRUPT_PENALTY * (DISRUPT_RATIO - change) / (DISRUPT_RATIO - 1.0)
        notes["disruption_penalty"] = round(penalty, 2)

    weights = active_weights()
    soft = sum(weights[k] * clamp(v, 0.0, 1.0) for k, v in bd.items())
    total = soft - HARD_PENALTY * len(hard) - penalty
    return Score(total=total, hard=hard, breakdown=bd, notes=notes, penalty=penalty)


def text_overflows(el: dict[str, Any], m: TextMeasurer, slack: float = OVERFLOW_SLACK) -> bool:
    """Does the *painted* text run past its box? The renderer never clips,
    so the box is a layout hint: what matters is where the glyphs land
    (widest wrapped line, last baseline plus descender) against the box,
    with the same half-pixel tolerance the designer DSL's validator uses
    (v6; the box-metric check also counted the last line's full descent).
    `max_lines` is still honoured. Shared with eval/metrics."""
    w, h = float(el["width"]), float(el["height"])
    font = float(el["font_size"])
    try:
        lay = m.layout(el, font, w)
        if el.get("max_lines") and lay.n_lines > int(el["max_lines"]):
            return True
        p = m.painted(el, font, max(w, 1.0))
    except Exception:
        return True
    return p.w > w + slack or p.bottom > h + slack


def bleed_allowance(unit: Unit, off_src: float, src_by_id: dict[str, dict[str, Any]]) -> float:
    """How much of a unit may lie off canvas: what it bled in the source
    plus BLEED_TOL; a picture / shape unit the designer already ran off
    the edge (v6) may bleed up to BLEED_GROW x that, capped at BLEED_CAP."""
    allowed = off_src + BLEED_TOL
    has_text = any(src_by_id.get(i, {}).get("type") == "text" for i in unit.ids)
    if not has_text and off_src >= BLEED_SRC_MIN:
        allowed = max(allowed, min(BLEED_GROW * off_src, BLEED_CAP))
    return allowed


def margin_judged(unit: Unit, src_by_id: dict[str, dict[str, Any]], src_canvas: Box) -> bool:
    """Does `layout.margins` look at this unit? Text and small pieces, yes;
    a picture or shape unit that is a block (>= BIG_UNIT_FRAC of the source
    canvas) is composition, and full-bleed is a legitimate choice for it."""
    if any(src_by_id.get(i, {}).get("type") == "text" for i in unit.ids):
        return True
    return unit.box.area < BIG_UNIT_FRAC * src_canvas.area


def below_floor(font: float, role: str, src_font: float, met: Metrics, area_scale: float) -> bool:
    """Is `font` too small for `role` on this canvas? Only when it is under
    the canvas's floor *and* smaller than the designer's own size scaled to
    the canvas: a 13 px label the source already had on a banner of this
    size is the designer's call, not a regression (shared with eval/metrics)."""
    if font >= met.min_font(role) - 0.5:
        return False
    return font < FLOOR_REL * src_font * area_scale - 0.5


def _primary_size(lives: list[Live], analysis: Analysis, by_id: dict[str, dict[str, Any]],
                  src_by_id: dict[str, dict[str, Any]], target: Box, area_scale: float) -> float:
    """How big the primary units came out, relative to AMBITION x the
    source-proportional size (v4): a headline or CTA label by its font, a
    hero or logo by its linear size on the canvas, each clamped at 1 so
    nothing is rewarded past the cap. 1.0 when the design has no primary
    unit. Unlike read.size this keeps paying above the proportional size,
    so among clean candidates the one that uses the room wins; margins,
    balance and the crowding penalty in fill keep it from going wall-to-wall."""
    num = den = 0.0
    for l in lives:
        role = l.unit.role
        w = SIZE_WEIGHTS.get(role)
        if w is None:
            continue
        texts = [e for e in l.unit.ids if by_id[e]["type"] == "text" and e in src_by_id]
        if role in (HEADLINE, CTA) and texts:
            # the unit's biggest text in the source (the label of a pill)
            eid = max(texts, key=lambda e: float(src_by_id[e]["font_size"]))
            src_font = float(src_by_id[eid]["font_size"])
            if src_font <= 0:
                continue
            ratio = float(by_id[eid]["font_size"]) / (src_font * area_scale)
        else:
            src_lin = math.sqrt(max(l.unit.box.area, 0.0) / analysis.canvas.area)
            if src_lin <= 0:
                continue
            ratio = math.sqrt(max(l.box.area, 0.0) / target.area) / src_lin
        num += w * clamp(ratio / AMBITION, 0.0, 1.0)
        den += w
    return num / den if den else 1.0


def _presence(lives: list[Live], src_canvas: Box, target: Box) -> float:
    """Hero and logo keep their visual weight: linear size relative to the
    canvas (sqrt of the area fraction) compared with the source, hero
    counting double. 1.0 when neither lost prominence."""
    num = den = 0.0
    for l in lives:
        if l.unit.role not in (HERO, LOGO):
            continue
        w = 2.0 if l.unit.role == HERO else 1.0
        src_lin = math.sqrt(max(l.unit.box.area, 0.0) / src_canvas.area)
        if src_lin <= 0:
            continue
        now_lin = math.sqrt(max(l.box.area, 0.0) / target.area)
        num += w * clamp(now_lin / src_lin, 0.0, 1.0)
        den += w
    return num / den if den else 1.0


# ---------------------------------------------------------------------------
# readability: contrast under text


def _luminance(rgb: tuple[float, float, float]) -> float:
    def ch(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _rgba(color: Any) -> tuple[tuple[int, int, int], float]:
    rgb = _rgb(color) or (0, 0, 0)
    alpha = 1.0
    if isinstance(color, str) and "(" in color:
        parts = [p.strip() for p in color[color.find("(") + 1: color.rfind(")")].split(",")]
        if len(parts) >= 4:
            try:
                alpha = clamp(float(parts[3]), 0.0, 1.0)
            except ValueError:
                alpha = 1.0
    return rgb, alpha


def _ink_boxes(elements: list[dict[str, Any]], m: TextMeasurer) -> dict[str, Box]:
    out = {}
    for el in elements:
        if el["type"] == "text" and str(el.get("content", "")).strip():
            out[el["id"]] = visible_box(el, Box.of(el), float(el["font_size"]), m)
    return out


def source_contrast(analysis: Analysis, source: dict[str, Any], m: TextMeasurer) -> dict[str, float]:
    """Estimated contrast of every text in the source design (cached)."""
    if analysis.source_contrast is None:
        est = contrast_est.estimate(source, _ink_boxes(source["elements"], m))
        analysis.source_contrast = {k: v.ratio for k, v in est.items()}
        analysis.source_straddle = {k for k, v in est.items() if is_straddle(v)}
    return analysis.source_contrast


def is_straddle(tc: "contrast_est.TextContrast") -> bool:
    """The text lies across a sharp edge: one slice reads badly, another
    well and several times better (engine/rectify.py has the figures)."""
    return _is_straddle_slices(tc.slices or [])


def straddling_texts(analysis: Analysis, source: dict[str, Any], elements: list[dict[str, Any]],
                     design: dict[str, Any], m: TextMeasurer) -> list[str]:
    """Ids of texts that straddle an edge in `design` but did not in the
    source (estimate only: no pixels needed). Shared with the rectifier."""
    est = contrast_est.estimate(design, _ink_boxes(elements, m))
    if not est:
        return []
    source_contrast(analysis, source, m)
    src = getattr(analysis, "source_straddle", None) or set()
    return [eid for eid, tc in est.items() if is_straddle(tc) and eid not in src]


def _contrast_terms(analysis: Analysis, source: dict[str, Any], elements: list[dict[str, Any]],
                    design: dict[str, Any], image, m: TextMeasurer
                    ) -> tuple[float, dict[str, float], dict[str, float], str, dict[str, Any]]:
    """(importance-weighted contrast score in [0, 1], per-text ratio, per-text
    source ratio, where the numbers came from, the JSON estimate itself).
    The JSON estimate serves every text; a render without text, when given,
    refines ratio and busyness for the texts it covers (the final top
    candidates)."""
    ratios, busy, est = _contrast_estimate(elements, design, m)
    how = "estimate" if ratios else "none"
    if image is not None:
        px_ratios, px_busy = _contrast_pixels(elements, design, image, m)
        if px_ratios:
            ratios.update(px_ratios)
            busy.update(px_busy)
            how = "pixels+estimate" if how == "estimate" else "pixels"
    src = source_contrast(analysis, source, m)
    if not ratios:
        return 1.0, {}, src, how, est
    num = den = 0.0
    for el in elements:
        eid = el["id"]
        if eid not in ratios:
            continue
        ratio = ratios[eid]
        b = clamp((busy.get(eid, 0.0) - 20.0) / 60.0, 0.0, 1.0)
        score = clamp((ratio - CONTRAST_FLOOR) / (CONTRAST_GOOD - CONTRAST_FLOOR), 0.0, 1.0) * (1.0 - 0.5 * b)
        imp = analysis.unit(analysis.element_unit[eid]).importance if eid in analysis.element_unit else 0.3
        imp = max(imp, 0.05)
        num += imp * score
        den += imp
    return (num / den if den else 1.0), ratios, src, how, est


def _contrast_estimate(elements: list[dict[str, Any]], design: dict[str, Any],
                       m: TextMeasurer) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    est = contrast_est.estimate(design, _ink_boxes(elements, m))
    return {k: v.ratio for k, v in est.items()}, {k: v.busy for k, v in est.items()}, est


def _backdrop_term(analysis: Analysis, source: dict[str, Any], elements: list[dict[str, Any]],
                   design: dict[str, Any], image, est: dict[str, Any], m: TextMeasurer
                   ) -> tuple[float, dict[str, Any]]:
    """v10 `read.backdrop` (engine/backdrop.py): per-text statistics under
    the headline / subhead / CTA ink from the pixels when `image` is given,
    else from the JSON contrast estimate already computed for the contrast
    term; the source measured the same way (cached per task)."""
    judged = {el["id"] for el in elements if el["type"] == "text" and analysis.roles.get(el["id"]) in backdrop.JUDGE_ROLES
              and str(el.get("content", "")).strip()}
    if not judged:
        return 1.0, {}
    if image is not None and not backdrop.pixels_enabled():
        image = None          # both stages judge the term on the composite (same direction, no source render)
    if image is not None:
        ink = {eid: box for eid, box in _ink_boxes(elements, m).items() if eid in judged}
        stats = backdrop.text_stats(design, ink, image, m)
    else:
        stats = {eid: (tc.mean_lum, tc.busy, tc.mean_rgb) for eid, tc in est.items() if eid in judged}
    if not stats:
        return 1.0, {}
    return backdrop.backdrop_term(analysis, source, design, stats, image, m)


def _contrast_pixels(elements: list[dict[str, Any]], design: dict[str, Any], image,
                     m: TextMeasurer) -> tuple[dict[str, float], dict[str, float]]:
    """Per-text contrast ratio and busyness from a render without text."""
    try:
        from PIL import ImageStat
    except Exception:  # pragma: no cover
        return {}, {}
    k = image_scale(design, image)
    ratios: dict[str, float] = {}
    busy: dict[str, float] = {}
    for el in elements:
        if el["type"] != "text" or not str(el.get("content", "")).strip():
            continue
        box = visible_box(el, Box.of(el), float(el["font_size"]), m)
        x0, y0 = int(max(0, math.floor(box.x * k))), int(max(0, math.floor(box.y * k)))
        x1, y1 = int(min(image.width, math.ceil(box.right * k))), int(min(image.height, math.ceil(box.bottom * k)))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image.crop((x0, y0, x1, y1))
        stat = ImageStat.Stat(crop)
        mean = tuple(float(v) for v in stat.mean[:3])
        rgb, alpha = _rgba(el.get("color"))
        blended = tuple(alpha * c + (1.0 - alpha) * b for c, b in zip(rgb, mean))
        ratios[el["id"]] = contrast_ratio(blended, mean)
        busy[el["id"]] = sum(float(v) for v in stat.stddev[:3]) / 3.0
    return ratios, busy


def _contrast(analysis: Analysis, elements: list[dict[str, Any]], design: dict[str, Any],
              image, m: TextMeasurer) -> tuple[float, dict[str, float]]:
    """Render-based contrast only (kept for eval/contrast_check.py)."""
    if image is None:
        return 1.0, {}
    ratios, busy = _contrast_pixels(elements, design, image, m)
    num = den = 0.0
    for eid, ratio in ratios.items():
        b = clamp((busy.get(eid, 0.0) - 20.0) / 60.0, 0.0, 1.0)
        score = clamp((ratio - CONTRAST_FLOOR) / (CONTRAST_GOOD - CONTRAST_FLOOR), 0.0, 1.0) * (1.0 - 0.5 * b)
        imp = analysis.unit(analysis.element_unit[eid]).importance if eid in analysis.element_unit else 0.3
        imp = max(imp, 0.05)
        num += imp * score
        den += imp
    return (num / den if den else 1.0), ratios


# ---------------------------------------------------------------------------
# layout quality (Ngo et al. style measures, each in [0, 1])


def _balance(lives: list[Live], target: Box, vis: dict[str, Box] | None = None) -> float:
    tw = tot = 0.0
    bx = by = 0.0
    for l in lives:
        b = vis[l.unit.key] if vis else l.box
        w = b.area * (l.importance + 0.1)
        bx += w * (b.cx - target.w / 2.0)
        by += w * (b.cy - target.h / 2.0)
        tot += w
    if tot <= 0:
        return 1.0
    bx = abs(bx / tot) / (target.w / 2.0)
    by = abs(by / tot) / (target.h / 2.0)
    return clamp(1.0 - (bx + by) / 2.0, 0.0, 1.0)


def _clusters(values: list[float], tol: float) -> int:
    values = sorted(values)
    n = 0
    last = None
    for v in values:
        if last is None or v - last > tol:
            n += 1
            last = v
    return n


def _alignment(boxes: list[Box], target: Box) -> float:
    n = len(boxes)
    if n < 2:
        return 1.0
    tol = max(2.0, 0.01 * min(target.w, target.h))
    xs = [v for b in boxes for v in (b.x, b.cx, b.right)]
    ys = [v for b in boxes for v in (b.y, b.cy, b.bottom)]
    cx, cy = _clusters(xs, tol), _clusters(ys, tol)
    # fewest possible distinct lines is 3 per axis (everything sharing left/centre/right)
    lines = clamp(1.0 - ((cx - 3) + (cy - 3)) / max(1.0, 6.0 * n - 6.0), 0.0, 1.0)
    aligned = 0
    for i, a in enumerate(boxes):
        ok = False
        for j, b in enumerate(boxes):
            if i == j:
                continue
            if (abs(a.x - b.x) <= tol or abs(a.cx - b.cx) <= tol or abs(a.right - b.right) <= tol
                    or abs(a.y - b.y) <= tol or abs(a.cy - b.cy) <= tol or abs(a.bottom - b.bottom) <= tol):
                ok = True
                break
        aligned += ok
    return 0.5 * lines + 0.5 * aligned / n


def _spacing_cvs(vis: list[Box]) -> list[float]:
    """Coefficient of variation of the gaps along every stack / row chain of
    three or more boxes (0 = perfectly even)."""
    cvs = []
    for axis in ("y", "x"):
        for chain in _chains(vis, axis):
            if len(chain) < 3:
                continue
            if axis == "y":
                gaps = [chain[i + 1].y - chain[i].bottom for i in range(len(chain) - 1)]
            else:
                gaps = [chain[i + 1].x - chain[i].right for i in range(len(chain) - 1)]
            mean = sum(gaps) / len(gaps)
            if mean <= 0:
                continue
            sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
            cvs.append(sd / mean)
    return cvs


def _spacing(vis: list[Box], free: float = 0.0) -> float:
    """Even gaps in stacks and rows; `free` is the unevenness (gap
    coefficient of variation) the source had, which costs nothing (v6)."""
    scores = [clamp(1.0 - max(0.0, cv - free), 0.0, 1.0) for cv in _spacing_cvs(vis)]
    return sum(scores) / len(scores) if scores else 1.0


def source_spacing_cv(analysis: Analysis, m: TextMeasurer) -> float:
    """Mean gap unevenness of the source's own stacks / rows (visible boxes,
    fine print left out like in the output), cached on the analysis."""
    cached = getattr(analysis, "_source_spacing_cv", None)
    if cached is not None:
        return cached
    by_id = analysis.source_elements
    try:
        lives = _lives(analysis, by_id)
        vis = [unit_visible_box(l, by_id, m) for l in lives if l.unit.role != FINE_PRINT]
        cvs = _spacing_cvs(vis)
        cv = sum(cvs) / len(cvs) if cvs else 0.0
    except Exception:  # pragma: no cover - odd source element
        cv = 0.0
    analysis._source_spacing_cv = cv   # type: ignore[attr-defined]
    return cv


def _chains(boxes: list[Box], axis: str) -> list[list[Box]]:
    order = sorted(boxes, key=lambda b: b.y if axis == "y" else b.x)
    nxt: dict[int, int] = {}
    taken: set[int] = set()
    for i, a in enumerate(order):
        best = None
        for j in range(i + 1, len(order)):
            b = order[j]
            if axis == "y":
                gap = b.y - a.bottom
                ov = min(a.right, b.right) - max(a.x, b.x)
                span = min(a.w, b.w)
            else:
                gap = b.x - a.right
                ov = min(a.bottom, b.bottom) - max(a.y, b.y)
                span = min(a.h, b.h)
            # Same row/column if they overlap at all on the other axis (a badge
            # centred on a banner row overlaps neither neighbour by half).
            if gap < -0.5 or ov < min(1.0, 0.5 * span):
                continue
            if best is None or gap < best[0]:
                best = (gap, j)
        if best is not None and best[1] not in taken:
            nxt[i] = best[1]
            taken.add(best[1])
    chains = []
    for i in range(len(order)):
        if i in taken:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        if len(chain) >= 2:
            chains.append([order[j] for j in chain])
    return chains


def _fill(boxes: list[Box], target: Box) -> float:
    if not boxes:
        return 0.0
    hull = Box.union(boxes)
    hull = Box(max(0.0, hull.x), max(0.0, hull.y), min(target.w, hull.right) - max(0.0, hull.x),
               min(target.h, hull.bottom) - max(0.0, hull.y))
    fill = clamp(hull.area / target.area / GOOD_FILL, 0.0, 1.0)
    density = sum(b.area for b in boxes) / target.area
    # v4: the hull alone saturates for a thin row across a banner; the ink
    # density says whether the content actually uses the canvas.
    used = clamp(density / GOOD_DENSITY, 0.0, 1.0)
    crowded = clamp((density - 0.7) / 0.3, 0.0, 1.0)
    return ((1.0 - DENSITY_SHARE) * fill + DENSITY_SHARE * used) * (1.0 - 0.5 * crowded)


def _margins(boxes: list[Box], target: Box) -> float:
    if not boxes:
        return 1.0
    mn = min(target.w, target.h)
    d = min(min(b.x, b.y, target.w - b.right, target.h - b.bottom) for b in boxes)
    return clamp(d / (0.03 * mn), 0.0, 1.0)


def _cohesion(boxes: list[Box], target: Box) -> float:
    """v5: the content holds together. Along each axis the boxes' projections
    are merged into runs; the widest empty stretch between runs may be
    max(COHESION_GAP_MULT x the base gap, COHESION_GAP_FRAC x the span)
    for free and costs the full term at half the span. A column with a
    bottom-anchored legal line is not judged here (fine print is left out
    by the caller); a row spread to the banner's ends is."""
    if len(boxes) < 2:
        return 1.0
    base_gap = clamp(0.035 * min(target.w, target.h), 4.0, 48.0)
    score = 1.0
    for axis in ("x", "y"):
        span = target.w if axis == "x" else target.h
        iv = sorted((b.x, b.right) if axis == "x" else (b.y, b.bottom) for b in boxes)
        runs = [list(iv[0])]
        for lo, hi in iv[1:]:
            if lo <= runs[-1][1]:
                runs[-1][1] = max(runs[-1][1], hi)
            else:
                runs.append([lo, hi])
        if len(runs) < 2:
            continue
        widest = max(runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1))
        allowed = max(COHESION_GAP_MULT * base_gap, COHESION_GAP_FRAC * span)
        score = min(score, 1.0 - clamp((widest - allowed) / (0.5 * span), 0.0, 1.0))
    return score


# ---------------------------------------------------------------------------
# faithfulness to the source


def reading_order(keys_boxes: list[tuple[str, Box]]) -> dict[str, int]:
    """Reading sequence of a laid-out design: rows are groups of boxes that
    overlap vertically (transitively, so a tall hero ties a banner row
    together), ordered top to bottom; inside a row, boxes that overlap
    horizontally form a stack (a headline over its subhead) ordered top to
    bottom, and stacks run left to right."""
    n = len(keys_boxes)
    boxes = [b for _, b in keys_boxes]

    def groups(idx: list[int], same) -> list[list[int]]:
        parent = {i: i for i in idx}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if same(boxes[i], boxes[j]):
                    parent[find(i)] = find(j)
        out: dict[int, list[int]] = {}
        for i in idx:
            out.setdefault(find(i), []).append(i)
        return list(out.values())

    def y_overlap(a: Box, b: Box) -> bool:
        return min(a.bottom, b.bottom) - max(a.y, b.y) >= 1.0

    def x_overlap(a: Box, b: Box) -> bool:
        return min(a.right, b.right) - max(a.x, b.x) >= 0.5 * max(1.0, min(a.w, b.w))

    order: dict[str, int] = {}
    pos = 0
    rows = groups(list(range(n)), y_overlap)
    rows.sort(key=lambda g: sum(boxes[i].cy for i in g) / len(g))
    for row in rows:
        stacks = groups(row, x_overlap)
        stacks.sort(key=lambda g: sum(boxes[i].cx for i in g) / len(g))
        for stack in stacks:
            for i in sorted(stack, key=lambda i: (boxes[i].cy, boxes[i].cx)):
                order[keys_boxes[i][0]] = pos
                pos += 1
    return order


def _order(lives: list[Live], vis: dict[str, Box]) -> float:
    if len(lives) < 2:
        return 1.0
    now = reading_order([(l.unit.key, vis[l.unit.key]) for l in lives])
    conc = disc = 0
    for i, a in enumerate(lives):
        for b in lives[i + 1:]:
            s = (a.unit.reading_index - b.unit.reading_index)
            t = (now[a.unit.key] - now[b.unit.key])
            if s * t > 0:
                conc += 1
            elif s * t < 0:
                disc += 1
    pairs = conc + disc
    if pairs == 0:
        return 1.0
    tau = (conc - disc) / pairs
    return (tau + 1.0) / 2.0


def _anchor_keep(lives: list[Live], target: Box) -> float:
    num = den = 0.0
    for l in lives:
        ax, ay = _anchors(l.box, target)
        u = l.unit
        w = max(u.importance, 0.05)
        if u.anchor_x != "fluid":
            den += w
            num += w * (ax == u.anchor_x)
        if u.anchor_y != "fluid":
            den += w
            num += w * (ay == u.anchor_y)
    return num / den if den else 1.0


def _hierarchy(elements: list[dict[str, Any]], src_by_id: dict[str, dict[str, Any]],
               roles: dict[str, str] | None = None) -> float:
    """Size relations kept: a text that was clearly bigger than another
    stays at least as big; a higher-ranked role (headline over subhead over
    body over badge over legal) keeps at least min(HIER_RATIO, the source's
    own ratio) of the lower one's size; a bigger image stays bigger."""
    roles = roles or {}
    texts = [el for el in elements if el["type"] == "text" and el["id"] in src_by_id]
    images = [el for el in elements if el["type"] == "image" and el["id"] in src_by_id]
    ok = tot = 0
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            sa, sb = float(src_by_id[a["id"]]["font_size"]), float(src_by_id[b["id"]]["font_size"])
            if sa >= 1.1 * sb:
                tot += 1
                ok += float(a["font_size"]) >= float(b["font_size"]) - 0.5
            elif sb >= 1.1 * sa:
                tot += 1
                ok += float(b["font_size"]) >= float(a["font_size"]) - 0.5
            ra, rb = RANK.get(roles.get(a["id"], ""), 2), RANK.get(roles.get(b["id"], ""), 2)
            if ra == rb:
                continue
            hi, lo = (a, b) if ra > rb else (b, a)
            s_hi, s_lo = float(src_by_id[hi["id"]]["font_size"]), float(src_by_id[lo["id"]]["font_size"])
            src_ratio = s_hi / max(1.0, s_lo)
            out_ratio = float(hi["font_size"]) / max(1.0, float(lo["font_size"]))
            tot += 1
            ok += out_ratio >= min(HIER_RATIO, src_ratio) - 0.01
    for i, a in enumerate(images):
        for b in images[i + 1:]:
            sa, sb = Box.of(src_by_id[a["id"]]).area, Box.of(src_by_id[b["id"]]).area
            if sa >= 1.5 * sb:
                tot += 1
                ok += Box.of(a).area >= Box.of(b).area
            elif sb >= 1.5 * sa:
                tot += 1
                ok += Box.of(b).area >= Box.of(a).area
    return ok / tot if tot else 1.0


def structure_survives(sc: "Score") -> bool:
    """v8: does this (structure) candidate's score say the source's
    composition survived the shape change? No hard fail, and the readability
    block (text size vs the canvas, headline / CTA line counts, hero / logo
    presence) at least STRUCTURE_READ_MIN on every term."""
    bd = sc.breakdown
    return not sc.hard and min(bd.get(k, 1.0) for k in SURVIVAL_TERMS) >= STRUCTURE_READ_MIN


def structure_prior(strategy: str | None, change: float, clean_structure: bool, survives: bool = False) -> float:
    """v8: the penalty a candidate pays because a surviving preserve-structure
    candidate (`structure_survives`) exists for this task: on a big shape
    change the template (floor plan) layouts and the structure variants that
    did not survive (`survives` is this candidate's own verdict) pay
    STRUCTURE_PRIOR; survivors and anchors (which carry the adaptation
    penalty) pay nothing. The caller (engine/resize.py) knows the candidate
    set; this is the rule."""
    if not clean_structure or change < ADAPT_RATIO:
        return 0.0
    if strategy == "floorplan" or (strategy == "structure" and not survives):
        return STRUCTURE_PRIOR
    return 0.0


def _line_stats(parts: list[Box], axis: str) -> tuple[float, list[float], list[float]]:
    """Parts sorted along `axis`: (share of consecutive pairs that overlap on
    the cross axis by ROW_KEEP_OVERLAP of the smaller extent, the gaps along
    the axis as a share of the larger neighbour's cross extent, the cross
    extents)."""
    if axis == "x":
        order = sorted(parts, key=lambda b: b.x)
        cross = [b.h for b in order]
        ov = [min(a.bottom, b.bottom) - max(a.y, b.y) for a, b in zip(order, order[1:])]
        gaps = [b.x - a.right for a, b in zip(order, order[1:])]
    else:
        order = sorted(parts, key=lambda b: b.y)
        cross = [b.w for b in order]
        ov = [min(a.right, b.right) - max(a.x, b.x) for a, b in zip(order, order[1:])]
        gaps = [b.y - a.bottom for a, b in zip(order, order[1:])]
    pairs = len(order) - 1
    inline = sum(o >= ROW_KEEP_OVERLAP * max(1.0, min(ca, cb)) for o, ca, cb in zip(ov, cross, cross[1:])) / pairs
    rel = [g / max(1.0, max(ca, cb)) for g, ca, cb in zip(gaps, cross, cross[1:])]
    return inline, rel, cross


def _cv(xs: list[float]) -> float:
    xs = [max(0.0, x) for x in xs]
    mean = sum(xs) / len(xs) if xs else 0.0
    if mean <= 0:
        return 0.0
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)) / mean


def row_shape(parts: list[Box]) -> dict[str, float]:
    """The shape of a line of parts: the better of the row and the column
    reading. `inline` is the share of consecutive pairs on one line, `gap`
    the widest gap relative to its neighbours' cross extent, `spread` the
    cross-extent spread (max - min) / max, `cv` the gap unevenness."""
    best = None
    thick = max(1.0, max(min(b.w, b.h) for b in parts))
    nn = max(min(_box_gap(a, b) for b in parts if b is not a) for a in parts) / thick
    for axis in ("x", "y"):
        inline, rel, cross = _line_stats(parts, axis)
        d = {"axis": 0.0 if axis == "x" else 1.0, "inline": inline, "gap": max(rel) if rel else 0.0,
             "spread": (max(cross) - min(cross)) / max(1e-6, max(cross)), "cv": _cv(rel) if len(rel) >= 2 else 0.0,
             "nn": nn}
        if best is None or (d["inline"], -d["gap"]) > (best["inline"], -best["gap"]):
            best = d
    return best or {"axis": 0.0, "inline": 1.0, "gap": 0.0, "spread": 0.0, "cv": 0.0, "nn": 0.0}


def _box_gap(a: Box, b: Box) -> float:
    """Chebyshev gap between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a.x, b.x) - min(a.right, b.right))
    dy = max(0.0, max(a.y, b.y) - min(a.bottom, b.bottom))
    return max(dx, dy)


def source_row_shapes(analysis: Analysis, m: TextMeasurer) -> dict[str, dict[str, float]]:
    """`row_shape` of every sibling row in the source (visible parts), cached
    on the analysis: what the designer's own row looked like is the baseline."""
    cached = getattr(analysis, "_source_row_shapes", None)
    if cached is not None:
        return cached
    out: dict[str, dict[str, float]] = {}
    by_id = analysis.source_elements
    for u in analysis.units:
        if not u.is_row or len(u.parts) < 2:
            continue
        try:
            live = Live(u, {eid: u.members[eid] for eid in u.ids if eid in by_id},
                        {eid: float(by_id[eid]["font_size"]) for eid in u.ids if eid in by_id and by_id[eid]["type"] == "text"})
            parts = unit_visible_parts(live, by_id, m)
        except Exception:  # pragma: no cover - odd source element
            parts = [p.box for p in u.parts]
        if len(parts) >= 2:
            out[u.key] = row_shape(parts)
    analysis._source_row_shapes = out   # type: ignore[attr-defined]
    return out


def _row_integrity(lives: list[Live], analysis: Analysis, by_id: dict[str, dict[str, Any]],
                   m: TextMeasurer) -> float:
    """v8: every sibling row of the source is still one line of parts (see
    ROW_KEEP_*), importance-weighted; 1.0 when the design has no rows. Each
    row scores max(inline x gap x evenness, cluster) x spread, every
    factor graded and source-relative (the designer's own row costs nothing)."""
    src = source_row_shapes(analysis, m)
    num = den = 0.0
    for l in lives:
        u = l.unit
        if not u.is_row or u.key not in src:
            continue
        parts = unit_visible_parts(l, by_id, m)
        if len(parts) < 2:
            continue
        s0 = src[u.key]
        now = row_shape(parts)
        inline = now["inline"]
        gap_allowed = max(ROW_KEEP_GAP, ROW_KEEP_GROW * s0["gap"])
        gap = clamp(1.0 - (now["gap"] - gap_allowed) / gap_allowed, 0.0, 1.0)
        spread_allowed = max(ROW_KEEP_SPREAD, s0["spread"] + 0.05)
        spread = clamp(1.0 - (now["spread"] - spread_allowed) / spread_allowed, 0.0, 1.0)
        even = clamp(1.0 - max(0.0, now["cv"] - s0["cv"] - ROW_KEEP_CV) / 0.5, 0.0, 1.0) if len(parts) >= 3 else 1.0
        nn_allowed = max(gap_allowed, ROW_KEEP_GROW * s0["nn"])
        cluster = clamp(1.0 - (now["nn"] - nn_allowed) / nn_allowed, 0.0, 1.0)
        w = max(u.importance, 0.05)
        num += w * max(inline * gap * even, cluster) * spread
        den += w
    return num / den if den else 1.0


def _scale_consistency(lives: list[Live]) -> float:
    scales = []
    for l in lives:
        sb = l.unit.box
        if sb.area <= 0:
            continue
        scales.append(math.sqrt(l.box.area / sb.area))
    if len(scales) < 2:
        return 1.0
    mean = sum(scales) / len(scales)
    if mean <= 0:
        return 0.0
    sd = math.sqrt(sum((s - mean) ** 2 for s in scales) / len(scales))
    return 1.0 / (1.0 + 2.0 * sd / mean)
