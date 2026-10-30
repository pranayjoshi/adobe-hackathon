"""Read the invisible rules out of a source design.

Hidden tasks give us only id / type / geometry, so every role here is
inferred from signals (size, position, opacity, grouping, font size,
text length, colour), never from element ids.

Output is an `Analysis`: a list of layout `Unit`s, each tagged with a role,
an importance weight, and the canvas edges it hugs. A unit is one piece of
furniture that travels through every layout as one thing. Beyond the
schema's `group_id`, units are found by union-find over these relations:

  holder    a text that sits (>= 80 %) inside a small flat shape or a flat
            image drawn below it: a label in its pill, a date in its square.
            The tightest holder wins. Such a unit scales uniformly (group).
  frame     a photo inside a shape of about its own size: the frame travels
            with the photo and the unit keeps the photo's role (hero/logo).
  host      a small shape that sits on, or just above / below, a bigger
            unit (an underline bar under a heading, a divider line, a
            sticker on a photo) is *attached* to it: the host keeps its
            own role and the attachment keeps its position relative to
            the host's painted extent when the host is re-fitted.
  absorbed  a decoration drawn >= 80 % inside a unit is attached to it.
  row       two or more chip-like units of the same role, similar height,
            side by side become one re-flowable row (`Unit.parts`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.geometry import Box

# Roles, roughly in importance order. Keep these strings stable: the layout
# code and the tests key off them.
BACKGROUND = "background"
DECORATIVE = "decorative"
LOGO = "logo"
HERO = "hero"
IMAGE = "image"
HEADLINE = "headline"
SUBHEAD = "subhead"
BODY = "body"
CTA = "cta"
BADGE = "badge"
FINE_PRINT = "fine_print"
PANEL = "panel"              # a large flat shape: a colour zone / card / bar, not a sticker
SHAPE = "shape"
TEXT = "text"

IMPORTANCE = {
    HEADLINE: 1.0,
    LOGO: 0.9,
    CTA: 0.9,
    HERO: 0.85,
    SUBHEAD: 0.6,
    BADGE: 0.5,
    BODY: 0.5,
    IMAGE: 0.4,
    SHAPE: 0.3,
    FINE_PRINT: 0.2,
    PANEL: 0.15,
    DECORATIVE: 0.1,
    BACKGROUND: 0.0,
    TEXT: 0.4,
}

# Thresholds (fractions of the canvas unless noted).
BACKGROUND_COVER = 0.85      # element covers >= this much of the canvas
DECO_OPACITY = 0.35          # below this an ungrouped shape is decoration
DECO_BLEED = 0.10            # >= this fraction of its own area off-canvas
LOGO_MAX_AREA = 0.15         # logo candidates are small
HERO_MIN_AREA = 0.03         # anything smaller is a sticker, not the hero
EDGE_HUG = 0.15              # margin < this fraction => anchored to that edge
CENTER_TOL_X = 0.06
CENTER_TOL_Y = 0.08
FINE_PRINT_RATIO = 0.45      # font_size <= this * headline size ...
FINE_PRINT_MAX_PX = 16       # ... and absolutely small, or very long (legal copy)
FINE_PRINT_MIN_CHARS = 80
CTA_MAX_CHARS = 32
PHRASE_ROW_TOL = 0.35       # v11 compound phrases: centres within this share of the smaller height ...
PHRASE_GAP_EM = 0.6         # ... at most this many ems apart ...
PHRASE_MAX_CHARS = 30       # ... and short together
PANEL_MIN_COVER = 0.15       # an ungrouped shape covering >= this much of the canvas is a panel ...
PANEL_EDGE = 0.02            # ... or one that runs from one canvas edge to the opposite one
PANEL_CARRY = 0.5            # a unit sits "on" a panel when this fraction of its box is inside it
PANEL_TEXT_CONTRAST = 1.5    # text with less contrast against the panel's fill is not carried by it

# Units beyond group_id.
HOLDER_MAX_AREA = 0.12       # a shape / flat image under this share of the canvas can hold one text ...
STICKER_MAX_AREA = 0.06      # ... or several short texts when it is sticker-sized (a date block);
STICKER_MAX_CHARS = 24       # ... (each text at most this long). A bigger shape holding text is a panel.
CARD_MAX_AREA = 0.25         # a shape up to this share holding text it cannot keep as a unit becomes a panel
INSIDE_FRAC = 0.80           # member inside its holder / host / unit by at least this much
FRAME_MAX_RATIO = 1.6        # photo-in-frame: the frame's area is at most this x the photo's
HOST_MAX_AREA = 0.03         # an attachable shape is at most this share of the canvas ...
HOST_MAX_REL = 0.6           # ... and at most this fraction of its host's area ...
HOST_THIN = 4.0              # ... and a bar (aspect >= this) unless it lies inside the host's box
HOST_GAP = 0.06              # ... and at most this x min(W, H) away from the host
HOST_X_OVERLAP = 0.5         # ... with at least this share of its width over the host's width
HOST_MAX_ATTACH = 3          # at most this many attachments per host
CTA_MIN_ASPECT = 1.0         # a button is at least as wide as it is tall and has one label
ROW_HEIGHT_TOL = 0.25        # sibling rows: heights within this fraction of each other
ROW_Y_OVERLAP = 0.6          # ... overlapping vertically by this share of the smaller height
ROW_MAX_GAP = 1.5            # ... horizontal gap at most this x the taller height
ROW_ROLES = (BADGE, IMAGE, SHAPE)   # chip-like roles that form rows
PANEL_STACK_SIZE = 0.7       # layered panels (v5): areas within this ratio of each other ...
PANEL_STACK_INSIDE = 0.6     # ... the smaller at least this much inside the larger ...
PANEL_STACK_TIGHT = 0.85     # ... and the same fill colour, or this much inside (a card and its shadow)


@dataclass
class Unit:
    """One piece of furniture: a single element, a schema group, an inferred
    cluster (holder / frame), a host with attachments, or a sibling row."""

    key: str                       # element id, "group:<group_id>", "unit:<id>" or "row:<id>"
    ids: list[str]
    role: str
    box: Box                       # bbox in source canvas px
    importance: float = 0.0
    anchor_x: str = "fluid"        # left | center | right | fluid
    anchor_y: str = "fluid"        # top | center | bottom | fluid
    reading_index: int = -1        # order among content units (y then x)
    group_id: str | None = None
    members: dict[str, Box] = field(default_factory=dict)   # id -> source box
    panel: str | None = None       # key of the panel this unit sits on in the source
    kind: str = "single"           # single | group | host | row
    primary: str | None = None     # the element the role comes from (single / host units)
    attachments: list[str] = field(default_factory=list)   # host units: the attached shapes / decorations
    parts: list["Unit"] = field(default_factory=list)       # row units: the sibling units, left to right

    @property
    def is_group(self) -> bool:
        """More than one element: moves and scales as one piece."""
        return len(self.ids) > 1

    @property
    def is_content(self) -> bool:
        """Furniture the plans arrange. Backgrounds, decoration and panels are
        zones laid out around the content instead."""
        return self.role not in (BACKGROUND, DECORATIVE, PANEL)

    @property
    def is_panel(self) -> bool:
        return self.role == PANEL

    @property
    def is_row(self) -> bool:
        return self.kind == "row"

    def primary_type(self, elements: dict[str, dict[str, Any]]) -> str | None:
        """Type of the element that defines this unit (None for groups / rows)."""
        if self.primary is None:
            return None
        return elements[self.primary]["type"]


@dataclass
class Analysis:
    canvas: Box
    units: list[Unit]
    roles: dict[str, str]          # element id -> role
    element_unit: dict[str, str]   # element id -> unit key
    headline_font_size: float | None = None
    shape_types: dict[str, str] = field(default_factory=dict)   # shape id -> shape_type
    source_contrast: dict[str, Any] | None = None   # filled lazily by the scorer (engine/contrast.py)
    source_straddle: set[str] | None = None         # texts that straddle an edge in the source (scorer fills it)
    source_elements: dict[str, dict[str, Any]] = field(default_factory=dict)   # id -> source element
    source_visible: dict[str, list[Box]] | None = None   # unit key -> painted part boxes in the source (rectify fills it)
    perception: dict[str, Any] | None = None        # S1: archetype + per-unit role votes (engine/role_merge.apply_votes fills it)
    patterns: list[Any] = field(default_factory=list)   # v10: arrangements of the free decorations (engine/patterns.Pattern)

    def unit(self, key: str) -> Unit:
        for u in self.units:
            if u.key == key:
                return u
        raise KeyError(key)

    def by_role(self, role: str) -> list[Unit]:
        return [u for u in self.units if u.role == role]

    def content_units(self) -> list[Unit]:
        return [u for u in self.units if u.is_content]

    def panels(self) -> list[Unit]:
        return [u for u in self.units if u.is_panel]

    def carried(self, panel_key: str) -> list[Unit]:
        """Content units that sat on this panel in the source."""
        return [u for u in self.units if u.is_content and u.panel == panel_key]


# ---------------------------------------------------------------------------


class _Cluster:
    """A union-find cluster while units are being built."""

    def __init__(self, eid: str):
        self.ids: list[str] = [eid]
        self.primary: str | None = eid
        self.attachments: list[str] = []
        self.group_id: str | None = None
        self.kind = "single"          # single | group | host


def analyze(source: dict[str, Any]) -> Analysis:
    canvas = Box(0.0, 0.0, float(source["canvas"]["width"]), float(source["canvas"]["height"]))
    elements = source["elements"]
    by_id = {el["id"]: el for el in elements}
    boxes = {el["id"]: Box.of(el) for el in elements}
    z = {el["id"]: float(el.get("z_index", i) if el.get("z_index") is not None else i)
         for i, el in enumerate(elements)}

    # 1. Per-element roles for things that are decided individually.
    roles: dict[str, str] = {}
    for el in elements:
        if _is_background(el, canvas):
            roles[el["id"]] = BACKGROUND
        elif _is_panel(el, canvas) and not _frames_image(el, elements, boxes, z):
            roles[el["id"]] = PANEL
        elif _is_decorative(el, canvas):
            roles[el["id"]] = DECORATIVE
    free = [el for el in elements if el["id"] not in roles]
    # 1b. Layered panels (v5): two colour zones of about the same size lying
    #     on each other (an offset double card, a card with its shadow) are
    #     one zone that keeps its layering; re-laying them out as two
    #     independent panels turned the offset into a stray bar.
    stacks = _panel_stacks([el for el in elements if roles.get(el["id"]) == PANEL], boxes, z)

    # 2. Union-find clusters: schema groups, holders, frames, hosts, decorations.
    clusters = _cluster(free, by_id, boxes, z, canvas, roles)

    # 3. Roles. Clusters with a primary element are classified like a free
    #    element of that type; the rest like a schema group.
    free_images = [by_id[c.primary] for c in clusters
                   if c.primary is not None and by_id[c.primary]["type"] == "image"]
    _assign_image_roles(free_images, canvas, roles)
    free_text = [by_id[c.primary] for c in clusters
                 if c.primary is not None and by_id[c.primary]["type"] == "text"]
    headline_size = _assign_text_roles(free_text, canvas, roles)
    for c in clusters:
        if c.primary is not None and c.primary not in roles:
            roles[c.primary] = SHAPE if by_id[c.primary]["type"] == "shape" else TEXT
    headline_box = next((boxes[el["id"]] for el in free_text if roles.get(el["id"]) == HEADLINE), None)
    groups = {i: [by_id[e] for e in c.ids] for i, c in enumerate(clusters) if c.primary is None}
    group_roles = _assign_group_roles(groups, canvas, headline_box)
    # One call to action per design: a button (label in a pill) is the
    # stronger signal, so a free text that merely looked like one steps down.
    if any(r == CTA for r in group_roles.values()):
        for el in free_text:
            if roles.get(el["id"]) == CTA:
                roles[el["id"]] = BODY
    for i, c in enumerate(clusters):
        role = roles[c.primary] if c.primary is not None else group_roles[i]
        for eid in c.ids:
            roles[eid] = role

    # 4. Build units.
    units: list[Unit] = []
    element_unit: dict[str, str] = {}
    for ids in stacks:
        mem = {i: boxes[i] for i in ids}
        u = Unit(key=ids[0], ids=list(ids), role=PANEL, box=Box.union(mem.values()), members=mem,
                 kind="stack", primary=ids[0])
        units.append(u)
        for i in ids:
            element_unit[i] = u.key
    for el in elements:
        if el["id"] in roles and roles[el["id"]] in (BACKGROUND, PANEL, DECORATIVE) and el["id"] not in element_unit:
            # non-content: a unit of its own unless a cluster absorbed it
            if any(el["id"] in c.ids for c in clusters):
                continue
            u = Unit(key=el["id"], ids=[el["id"]], role=roles[el["id"]], box=boxes[el["id"]],
                     members={el["id"]: boxes[el["id"]]}, primary=el["id"])
            units.append(u)
            element_unit[el["id"]] = u.key
    for c in clusters:
        mem = {eid: boxes[eid] for eid in c.ids}
        if c.group_id is not None:
            key = f"group:{c.group_id}"
        elif len(c.ids) == 1:
            key = c.ids[0]
        elif c.primary is not None:
            key = c.primary
        else:
            key = f"unit:{c.ids[0]}"
        u = Unit(key=key, ids=list(c.ids), role=roles[c.ids[0]], box=Box.union(mem.values()),
                 group_id=c.group_id, members=mem, kind=c.kind, primary=c.primary,
                 attachments=list(c.attachments))
        units.append(u)
        for eid in c.ids:
            element_unit[eid] = u.key

    for u in units:
        u.importance = IMPORTANCE.get(u.role, 0.3)
        u.anchor_x, u.anchor_y = _anchors(u.box, canvas)

    # 5. Sibling rows of chip-like units.
    units = _merge_rows(units, canvas)
    for u in units:
        for eid in u.ids:
            element_unit[eid] = u.key

    # Which panel does each piece of content sit on? The *topmost* panel
    # drawn below it that holds most of the unit's box: that is the colour
    # the viewer sees behind it (content under an opaque panel is hidden, not
    # carried). The layouts keep that relation.
    panels = sorted((u for u in units if u.is_panel), key=lambda u: (-z[u.ids[0]], u.box.area))
    for u in units:
        if not u.is_content:
            continue
        top = min(z[i] for i in u.ids)
        for p in panels:
            if z[p.ids[0]] < top and p.box.contains_fraction(u.box) >= PANEL_CARRY \
                    and _readable_on(u, by_id[p.ids[0]], by_id):
                u.panel = p.key
                break

    # Reading order over content units: top-to-bottom, then left-to-right,
    # with a row tolerance so side-by-side badges count as one row.
    content = sorted((u for u in units if u.is_content), key=lambda u: (u.box.cy, u.box.cx))
    rows: list[list[Unit]] = []
    for u in content:
        if rows and abs(rows[-1][0].box.cy - u.box.cy) < 0.5 * min(rows[-1][0].box.h, u.box.h) + 1:
            rows[-1].append(u)
        else:
            rows.append([u])
    idx = 0
    for row in rows:
        for u in sorted(row, key=lambda u: u.box.cx):
            u.reading_index = idx
            idx += 1

    # Stable, meaningful order: by reading index, non-content last.
    units.sort(key=lambda u: (u.reading_index if u.reading_index >= 0 else 10_000, u.key))
    shape_types = {el["id"]: str(el.get("shape_type", "rectangle") or "rectangle")
                   for el in elements if el["type"] == "shape"}
    # v10: the free decorations as one arrangement (engine/patterns.py);
    # only read when backdrop awareness is on (relayout.decorative_boxes).
    from engine.patterns import detect_patterns   # local: keeps analyze import-light
    patterns = detect_patterns([u for u in units if u.role == DECORATIVE], canvas, by_id)
    return Analysis(canvas=canvas, units=units, roles=roles, element_unit=element_unit,
                    headline_font_size=headline_size, shape_types=shape_types, source_elements=by_id,
                    patterns=patterns)


# ---------------------------------------------------------------------------
# union-find over the relations that make several elements one unit


def _canon(boxes: dict[str, Box], z: dict[str, float]):
    """v8: an id-free, order-free ranking of elements - drawing order (z),
    then reading position (y, x), then size - so a unit's member order, its
    first element (the "primary" of a pill, the key of a group) and every
    tie in the clustering are the same whatever order the task lists the
    elements in. The id is the last resort, for elements identical in all
    of those (then the choice cannot change what is drawn)."""
    def key(eid: str) -> tuple:
        b = boxes[eid]
        return (z[eid], round(b.y, 3), round(b.x, 3), round(b.area, 3), round(b.w, 3), eid)
    return key


def _cluster(free: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], boxes: dict[str, Box],
             z: dict[str, float], canvas: Box, roles: dict[str, str]) -> list[_Cluster]:
    parent: dict[str, str] = {el["id"]: el["id"] for el in free}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    group_ids: dict[str, str] = {}
    kinds: dict[str, str] = {}        # root -> "group" once a holder / frame / schema group joined it
    primaries: dict[str, str] = {}    # root -> primary element of a host-kind cluster
    attachments: dict[str, list[str]] = {}

    # a. schema groups
    by_gid: dict[str, list[str]] = {}
    for el in free:
        gid = el.get("group_id")
        if gid:
            by_gid.setdefault(str(gid), []).append(el["id"])
    for gid, ids in by_gid.items():
        for other in ids[1:]:
            union(ids[0], other)
        group_ids[find(ids[0])] = gid
        kinds[find(ids[0])] = "group"
    grouped = {eid for ids in by_gid.values() for eid in ids}

    def is_holder(el: dict[str, Any]) -> bool:
        b = boxes[el["id"]]
        if b.area <= 0 or b.area / canvas.area >= CARD_MAX_AREA:
            return False
        if el["type"] == "shape":
            return str(el.get("shape_type", "rectangle") or "rectangle") != "line"
        if el["type"] == "image":
            from engine.contrast import asset_is_flat   # local: keeps analyze import-light
            return asset_is_flat(el.get("asset_id"))
        return False

    canon = _canon(boxes, z)
    free = sorted(free, key=lambda el: canon(el["id"]))   # copy, canonical order (card holders may be dropped below)
    holders = [el for el in free if el["id"] not in grouped and is_holder(el)]

    # b. holders: a text inside a small flat shape / image drawn below it.
    #    A holder keeps one label (a pill, a button); a sticker-sized one may
    #    keep several short lines (a date block). A card with paragraphs is
    #    a zone, not a unit: its texts stay free and keep their own floors.
    held: dict[str, list[dict[str, Any]]] = {}
    for el in free:
        if el["type"] != "text" or el["id"] in grouped:
            continue
        tb = boxes[el["id"]]
        if tb.area <= 0:
            continue
        best = None
        for h in holders:
            hb = boxes[h["id"]]
            if z[h["id"]] >= z[el["id"]] or hb.contains_fraction(tb) < INSIDE_FRAC:
                continue
            if best is None or hb.area < boxes[best["id"]].area:
                best = h
        if best is not None:
            held.setdefault(best["id"], []).append(el)
    for hid, texts in held.items():
        share = boxes[hid].area / canvas.area
        if len(texts) > 1:
            keep = share <= STICKER_MAX_AREA and all(
                len(str(t.get("content", "")).strip()) <= STICKER_MAX_CHARS for t in texts)
        else:
            keep = share <= HOLDER_MAX_AREA
        if not keep:
            # A card: a zone rebuilt around the text it carries (panel), so
            # the text keeps its background and its own font floors.
            if by_id[hid]["type"] == "shape":
                roles[hid] = PANEL
                parent.pop(hid, None)
            continue
        for t in texts:
            union(hid, t["id"])
            kinds[find(t["id"])] = "group"

    free = [el for el in free if el["id"] in parent]   # card holders became panels

    # c. photo-in-frame: an image and a shape of about the same size and place
    for el in free:
        if el["type"] != "image" or el["id"] in grouped:
            continue
        ib = boxes[el["id"]]
        if ib.area <= 0:
            continue
        for s in free:
            if s["type"] != "shape" or s["id"] in grouped:
                continue
            sb = boxes[s["id"]]
            if sb.area <= 0 or sb.area > FRAME_MAX_RATIO * ib.area:
                continue
            if sb.contains_fraction(ib) >= INSIDE_FRAC and ib.contains_fraction(sb) >= 0.6:
                union(el["id"], s["id"])
                r = find(el["id"])
                if kinds.get(r) != "group":
                    kinds[r] = "host"
                    primaries[r] = el["id"]
                    attachments.setdefault(r, []).append(s["id"])

    # d. hosts: a thin bar (or a shape inside the host's box) sitting on /
    #    just above / just below a bigger single element. Tested against
    #    the host element itself, never against what is already attached,
    #    so attachments never chain away from their host.
    def cluster_box(root: str) -> Box:
        return Box.union(boxes[i] for i in parent if find(i) == root)

    mn = min(canvas.w, canvas.h)
    small = [el for el in free if el["type"] == "shape" and el["id"] not in grouped
             and sum(1 for i in parent if find(i) == find(el["id"])) == 1
             and 0 < boxes[el["id"]].area <= HOST_MAX_AREA * canvas.area]
    small.sort(key=lambda el: (boxes[el["id"]].area, canon(el["id"])))
    n_attached: dict[str, int] = {}
    for el in small:
        sb = boxes[el["id"]]
        thin = max(sb.w, sb.h) >= HOST_THIN * max(1.0, min(sb.w, sb.h))
        best = None
        for root in sorted({find(i) for i in parent}, key=canon):
            if root == el["id"] or find(el["id"]) == root:
                continue
            if kinds.get(root, "single") == "group":
                continue          # a pill / sticker is complete; bars attach to free text and images
            host_id = primaries.get(root, root)
            if by_id[host_id]["type"] not in ("text", "image") or n_attached.get(root, 0) >= HOST_MAX_ATTACH:
                continue
            hb = boxes[host_id]
            if hb.area <= 0 or sb.area > HOST_MAX_REL * hb.area:
                continue
            inside = hb.contains_fraction(sb) >= INSIDE_FRAC
            if not thin and not inside:
                continue
            xo = min(sb.right, hb.right) - max(sb.x, hb.x)
            if xo < HOST_X_OVERLAP * sb.w:
                continue
            gap = max(0.0, max(sb.y - hb.bottom, hb.y - sb.bottom))
            if gap > HOST_GAP * mn:
                continue
            above = hb.cy < sb.cy
            key = (gap, 0 if above else 1, hb.area)
            if best is None or key < best[0]:
                best = (key, root)
        if best is None:
            continue
        root = best[1]
        union(root, el["id"])
        new_root = find(el["id"])
        kinds[new_root] = "host"
        primaries[new_root] = primaries.get(root, root)
        attachments[new_root] = attachments.pop(root, []) + [el["id"]]
        n_attached[new_root] = n_attached.pop(root, 0) + 1
        for k in (kinds, primaries, attachments, group_ids):
            if root in k and root != new_root:
                k[new_root] = k.pop(root)

    # d2. compound phrases (v11, engine/flags UNITS): two single-line texts on
    #     one baseline row, a word space apart ("JUNE" "12-18", "SUNDAY, SEP 8"
    #     "5PM"), read as one phrase: the larger (then longer) text hosts the
    #     other as an attachment, so the pair keeps its spacing in font units.
    from engine import flags   # local: keeps analyze import-light
    if flags.enabled("units"):
        singles = [el for el in free if el["type"] == "text" and el["id"] not in grouped
                   and sum(1 for i in parent if find(i) == find(el["id"])) == 1
                   and "\n" not in str(el.get("content", "")).strip()
                   and abs(float(el.get("rotation", 0.0) or 0.0)) < 1.0]
        singles.sort(key=lambda el: canon(el["id"]))
        for a in singles:
            for b in singles:
                if a is b or find(a["id"]) == find(b["id"]):
                    continue
                if sum(1 for i in parent if find(i) == find(b["id"])) > 1:
                    continue
                ab, bb = boxes[a["id"]], boxes[b["id"]]
                fa, fb = float(a["font_size"]), float(b["font_size"])
                fm = min(fa, fb)
                ca, cb = str(a.get("content", "")).strip(), str(b.get("content", "")).strip()
                gap = bb.x - ab.right
                if not (abs(ab.cy - bb.cy) <= PHRASE_ROW_TOL * min(ab.h, bb.h)
                        and -0.1 * fm <= gap <= PHRASE_GAP_EM * fm and len(ca) + len(cb) <= PHRASE_MAX_CHARS):
                    continue
                if sum(1 for i in parent if find(i) == find(a["id"])) > 1 and kinds.get(find(a["id"])) != "host":
                    continue
                host, guest = (a, b) if (fa, len(ca)) >= (fb, len(cb)) else (b, a)
                root = find(host["id"])
                union(root, guest["id"])
                new_root = find(host["id"])
                kinds[new_root] = "host"
                primaries[new_root] = primaries.get(root, host["id"])
                attachments[new_root] = attachments.pop(root, []) + [guest["id"]]
                break

    # e. decorations drawn inside a content unit travel with it
    roots = sorted({find(i) for i in parent}, key=canon)
    for eid, role in sorted(roles.items(), key=lambda kv: canon(kv[0])):
        if role != DECORATIVE:
            continue
        db = boxes[eid]
        if db.area <= 0:
            continue
        best = None
        for root in roots:
            rb = cluster_box(root)
            if rb.area < db.area or rb.contains_fraction(db) < INSIDE_FRAC:
                continue
            if best is None or rb.area < best[0]:
                best = (rb.area, root)
        if best is None:
            continue
        root = best[1]
        parent[eid] = root
        del roles[eid]
        if kinds.get(root, "single") != "group":
            kinds[root] = "host"
            primaries.setdefault(root, root)
        attachments.setdefault(root, []).append(eid)

    out: list[_Cluster] = []
    members: dict[str, list[str]] = {}
    for i in parent:
        members.setdefault(find(i), []).append(i)
    # Clusters and their members in the canonical order (v8): the first id is
    # the lowest-drawn, top-left-most member (a pill's background before its
    # label), whatever order the task listed the elements in.
    for root, ids in sorted(members.items(), key=lambda kv: min(canon(i) for i in kv[1])):
        c = _Cluster(root)
        c.ids = sorted(ids, key=canon)
        c.group_id = group_ids.get(root)
        c.kind = kinds.get(root, "single")
        if c.kind == "group":
            c.primary = None
        elif c.kind == "host":
            c.primary = primaries.get(root, root)
            c.attachments = [i for i in attachments.get(root, []) if i != c.primary]
            # a host cluster must have exactly one non-attached element
            if len(c.ids) - len(c.attachments) != 1:
                c.kind, c.primary, c.attachments = "group", None, []
        else:
            c.primary = c.ids[0]
        out.append(c)
    return out


def _panel_stacks(panels: list[dict[str, Any]], boxes: dict[str, Box], z: dict[str, float]) -> list[list[str]]:
    """Groups of rectangular panels that lie on each other at about the same
    size (ids top-most first). Pairs qualify when the smaller is at least
    PANEL_STACK_INSIDE inside the larger, their areas are within
    PANEL_STACK_SIZE, and either the fills match or the smaller is
    PANEL_STACK_TIGHT inside (a shadow in another colour)."""
    rects = [el for el in panels
             if str(el.get("shape_type", "rectangle") or "rectangle") == "rectangle" and boxes[el["id"]].area > 0]
    parent = {el["id"]: el["id"] for el in rects}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            ba, bb = boxes[a["id"]], boxes[b["id"]]
            small, big = (ba, bb) if ba.area <= bb.area else (bb, ba)
            if small.area / big.area < PANEL_STACK_SIZE:
                continue
            inside = big.contains_fraction(small)
            same = _rgb(a.get("fill_color")) == _rgb(b.get("fill_color"))
            if inside >= PANEL_STACK_INSIDE and (same or inside >= PANEL_STACK_TIGHT):
                parent[find(a["id"])] = find(b["id"])
    groups: dict[str, list[str]] = {}
    for el in rects:
        groups.setdefault(find(el["id"]), []).append(el["id"])
    canon = _canon(boxes, z)
    stacks = [sorted(ids, key=lambda i: (-z[i], canon(i))) for ids in groups.values() if len(ids) > 1]
    return sorted(stacks, key=lambda ids: canon(ids[-1]))


def _merge_rows(units: list[Unit], canvas: Box) -> list[Unit]:
    """Chains of chip-like units of one role, similar height, side by side
    (left to right) become one row unit with the chain as `parts`."""
    cand = [u for u in units if u.is_content and u.role in ROW_ROLES and not u.is_row]
    cand.sort(key=lambda u: (u.box.x, u.box.y, u.key))
    used: set[str] = set()
    rows: list[list[Unit]] = []
    for u in cand:
        if u.key in used:
            continue
        chain = [u]
        used.add(u.key)
        while True:
            last = chain[-1]
            nxt = None
            for v in cand:
                if v.key in used or v.role != last.role:
                    continue
                a, b = last.box, v.box
                if v.box.x < last.box.x:
                    continue
                hmin, hmax = min(a.h, b.h), max(a.h, b.h)
                if hmin <= 0 or (hmax - hmin) / hmax > ROW_HEIGHT_TOL:
                    continue
                yo = min(a.bottom, b.bottom) - max(a.y, b.y)
                if yo < ROW_Y_OVERLAP * hmin:
                    continue
                gap = b.x - a.right
                if gap < -0.5 * hmin or gap > ROW_MAX_GAP * hmax:
                    continue
                if nxt is None or gap < nxt[0]:
                    nxt = (gap, v)
            if nxt is None:
                break
            chain.append(nxt[1])
            used.add(nxt[1].key)
        if len(chain) >= 2:
            rows.append(chain)
    if not rows:
        return units
    merged: dict[str, Unit] = {}
    for chain in rows:
        mem = {eid: b for u in chain for eid, b in u.members.items()}
        row = Unit(key=f"row:{chain[0].key}", ids=[i for u in chain for i in u.ids], role=chain[0].role,
                   box=Box.union(mem.values()), members=mem, kind="row", parts=list(chain))
        row.importance = max(u.importance for u in chain)
        row.anchor_x, row.anchor_y = _anchors(row.box, canvas)
        for u in chain:
            merged[u.key] = row
    out: list[Unit] = []
    seen: set[str] = set()
    for u in units:
        r = merged.get(u.key)
        if r is None:
            out.append(u)
        elif r.key not in seen:
            seen.add(r.key)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# individual classifiers


def _is_background(el: dict[str, Any], canvas: Box) -> bool:
    if el["type"] == "text":
        return False
    box = Box.of(el)
    covered = canvas.intersection_area(box) / canvas.area
    if covered < BACKGROUND_COVER:
        return False
    # A huge, nearly transparent shape is a wash, which is still background-like.
    return True


def _is_panel(el: dict[str, Any], canvas: Box) -> bool:
    """A large flat shape: a colour block behind text, a side bar, a split
    background, a frame. It is a zone, not a sticker: layouts stretch it (flat
    colour has no aspect) and keep the content that sat on it on top."""
    if el["type"] != "shape" or el.get("group_id"):
        return False
    box = Box.of(el)
    if box.area <= 0:
        return False
    covered = canvas.intersection_area(box) / canvas.area
    rect = str(el.get("shape_type", "rectangle") or "rectangle") == "rectangle"
    if rect:
        spans_x = box.x <= PANEL_EDGE * canvas.w and box.right >= (1.0 - PANEL_EDGE) * canvas.w
        spans_y = box.y <= PANEL_EDGE * canvas.h and box.bottom >= (1.0 - PANEL_EDGE) * canvas.h
        return covered >= PANEL_MIN_COVER or spans_x or spans_y
    # A big ellipse is a panel only when it is solid and sits on the canvas;
    # a faint or bleeding one is decoration.
    opacity = float(el.get("opacity", 1.0))
    off = 1.0 - canvas.contains_fraction(box)
    return covered >= PANEL_MIN_COVER and opacity >= DECO_OPACITY and off < DECO_BLEED


def _frames_image(el: dict[str, Any], elements: list[dict[str, Any]], boxes: dict[str, Box],
                  z: dict[str, float]) -> bool:
    """A shape drawn right behind a photo of about its own size is that
    photo's frame (one unit with it), however big: not a colour zone."""
    if el["type"] != "shape" or el.get("group_id"):
        return False
    sb = boxes[el["id"]]
    for o in elements:
        if o["type"] != "image" or o.get("group_id") or z[o["id"]] <= z[el["id"]]:
            continue
        ib = boxes[o["id"]]
        if ib.area <= 0 or sb.area > FRAME_MAX_RATIO * ib.area:
            continue
        if sb.contains_fraction(ib) >= INSIDE_FRAC and ib.contains_fraction(sb) >= 0.6:
            return True
    return False


def _is_decorative(el: dict[str, Any], canvas: Box) -> bool:
    if el["type"] == "text":
        return False
    if el.get("group_id"):
        return False
    box = Box.of(el)
    opacity = float(el.get("opacity", 1.0))
    if opacity < DECO_OPACITY:
        return True
    if box.area > 0:
        off = 1.0 - canvas.contains_fraction(box)
        if off >= DECO_BLEED:
            return True
    return False


def _assign_image_roles(images: list[dict[str, Any]], canvas: Box, roles: dict[str, str]) -> None:
    if not images:
        return
    scored: list[tuple[float, dict[str, Any]]] = []
    for el in images:
        box = Box.of(el)
        area = box.area / canvas.area
        if area > LOGO_MAX_AREA:
            continue
        cy = box.cy / canvas.h
        cx = box.cx / canvas.w
        score = 0.0
        if cy < 0.35:
            score += 2.0
        if abs(cx - 0.5) < 0.1:
            score += 0.5
        if cx < 0.25 or cx > 0.75:
            score += 0.5
        score -= area * 5.0
        if cy > 0.6:
            score -= 2.0   # bottom-dwelling small images are stickers/watermarks
        scored.append((score, el))
    logo = None
    if scored:
        scored.sort(key=lambda t: -t[0])
        if scored[0][0] > 0:
            logo = scored[0][1]
            roles[logo["id"]] = LOGO
    rest = [el for el in images if logo is None or el["id"] != logo["id"]]
    if rest:
        rest.sort(key=lambda el: -Box.of(el).area)
        biggest = rest[0]
        if Box.of(biggest).area / canvas.area >= HERO_MIN_AREA:
            roles[biggest["id"]] = HERO
            rest = rest[1:]
        for el in rest:
            roles[el["id"]] = IMAGE


def _assign_text_roles(texts: list[dict[str, Any]], canvas: Box, roles: dict[str, str]) -> float | None:
    if not texts:
        return None
    ordered = sorted(texts, key=lambda el: -float(el["font_size"]))
    headline = ordered[0]
    roles[headline["id"]] = HEADLINE
    h_size = float(headline["font_size"])
    h_box = Box.of(headline)
    h_color = _rgb(headline.get("color"))

    # A free-text CTA: bold, short, below the headline, in a colour of its
    # own. Several lines can look like that (section labels in an accent
    # colour); a design has one call to action, and it is the lowest one.
    cta_like = [
        el for el in ordered[1:]
        if el.get("font_weight") == "bold" and len(str(el.get("content", ""))) <= CTA_MAX_CHARS
        and Box.of(el).cy > h_box.cy and _rgb(el.get("color")) != h_color
        and not _fine_print_like(el, h_size)
    ]
    cta_id = max(cta_like, key=lambda el: Box.of(el).cy)["id"] if cta_like else None

    subhead_taken = False
    for el in ordered[1:]:
        box = Box.of(el)
        if _fine_print_like(el, h_size):
            roles[el["id"]] = FINE_PRINT
        elif el["id"] == cta_id:
            roles[el["id"]] = CTA
        elif not subhead_taken and box.cy > h_box.cy:
            roles[el["id"]] = SUBHEAD
            subhead_taken = True
        else:
            roles[el["id"]] = BODY
    return h_size


def _fine_print_like(el: dict[str, Any], headline_size: float) -> bool:
    size = float(el["font_size"])
    content = str(el.get("content", ""))
    return size <= FINE_PRINT_RATIO * headline_size and (
        size <= FINE_PRINT_MAX_PX or len(content) >= FINE_PRINT_MIN_CHARS
    )


def _assign_group_roles(groups: dict[Any, list[dict[str, Any]]], canvas: Box,
                        headline_box: Box | None) -> dict[Any, str]:
    out: dict[Any, str] = {}
    button_like: list[tuple[Any, Box]] = []
    for gid, members in groups.items():
        types = {el["type"] for el in members}
        box = Box.union(Box.of(el) for el in members)
        if "text" in types and ("shape" in types or "image" in types):
            button_like.append((gid, box))
            out[gid] = BADGE
        elif types == {"text"}:
            out[gid] = BODY
        elif "image" in types:
            out[gid] = IMAGE
        else:
            out[gid] = SHAPE
    # The CTA is the largest button-like group that sits below the headline
    # (or, with no headline, in the lower half of the canvas): one label in
    # a pill at least as wide as it is tall. A tall card of several lines is
    # a sticker, not a button.
    candidates = []
    for gid, box in button_like:
        below = box.cy > headline_box.cy if headline_box else box.cy > canvas.h / 2
        labels = [el for el in groups[gid] if el["type"] == "text"]
        one_label = len(labels) == 1
        if one_label and _units_on():
            # v11: a button holds a short label; a paragraph in a box is a card
            text = str(labels[0].get("content", "")).strip()
            one_label = len(text) <= 2 * CTA_MAX_CHARS and text.count("\n") <= 1
        if below and one_label and box.w >= CTA_MIN_ASPECT * box.h:
            candidates.append((box.area, gid))
    if candidates:
        candidates.sort(key=lambda t: -t[0])
        out[candidates[0][1]] = CTA
    return out


def _units_on() -> bool:
    from engine import flags
    return flags.enabled("units")


def _readable_on(unit: Unit, panel: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> bool:
    """Text in (nearly) the panel's own colour was not meant to be read on
    it (white copy half over a white card in a loosely built template): the
    panel must not be rebuilt around it."""
    fill = _rgb(panel.get("fill_color"))
    if fill is None:
        return True
    texts = [by_id[i] for i in unit.ids if by_id[i]["type"] == "text"]
    if not texts:
        return True
    for el in texts:
        rgb = _rgb(el.get("color"))
        if rgb is None or contrast_ratio(rgb, fill) >= PANEL_TEXT_CONTRAST:
            return True
    return False


def _luminance(rgb: tuple[int, int, int]) -> float:
    def ch(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _anchors(box: Box, canvas: Box) -> tuple[str, str]:
    l, r = box.x / canvas.w, (canvas.w - box.right) / canvas.w
    t, b = box.y / canvas.h, (canvas.h - box.bottom) / canvas.h
    cx, cy = box.cx / canvas.w, box.cy / canvas.h
    if abs(cx - 0.5) < CENTER_TOL_X:
        ax = "center"
    elif l < EDGE_HUG and l <= r:
        ax = "left"
    elif r < EDGE_HUG:
        ax = "right"
    else:
        ax = "fluid"
    if t < EDGE_HUG and t <= b:
        ay = "top"
    elif b < EDGE_HUG:
        ay = "bottom"
    elif abs(cy - 0.5) < CENTER_TOL_Y:
        ay = "center"
    else:
        ay = "fluid"
    return ax, ay


def _rgb(color: Any) -> tuple[int, int, int] | None:
    if not isinstance(color, str):
        return None
    inner = color[color.find("(") + 1: color.rfind(")")]
    parts = [p.strip() for p in inner.split(",")]
    try:
        return tuple(int(float(p)) for p in parts[:3])  # type: ignore[return-value]
    except ValueError:
        return None
