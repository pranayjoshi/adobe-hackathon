"""Semantic analysis: roles, relationships and layout blocks.

The input carries no semantic labels, so roles are confidence-weighted
guesses from geometry, typography, layering and explicit `group_id`s. IDs are
never used as hints - unseen inputs will not have descriptive names.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .structure import flat_asset
from .blocks import Block, ButtonBlock, ImageBlock, RigidBlock, RowBlock, TextBlock
from .canonical import Box, asset_exists, element_box, visual_box

IMPORTANCE = {
    "headline": 1.0, "hero": 0.9, "cta": 0.85, "logo": 0.8, "subhead": 0.6, "image": 0.6,
    "badges": 0.5, "badge": 0.5, "body": 0.5, "group": 0.5, "sticker": 0.4, "legal": 0.3, "accent": 0.2,
}
LABELS = {
    "headline": "Headline", "hero": "Hero image", "cta": "Call to action", "logo": "Logo",
    "subhead": "Supporting copy", "body": "Body copy", "legal": "Fine print", "badge": "Badge",
    "badges": "Badge row", "sticker": "Sticker", "accent": "Accent", "image": "Image", "group": "Group",
    "background": "Background", "decoration": "Decoration",
}


@dataclass
class Analysis:
    sw: float
    sh: float
    background: list[dict[str, Any]] = field(default_factory=list)
    decorations: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)     # main content, reading order
    stickers: list[Block] = field(default_factory=list)   # rotated / corner-anchored units
    accents: list[Block] = field(default_factory=list)    # small or missing-asset images
    roles: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    panels: list[dict[str,Any]] = field(default_factory=list)
    relationships: list[dict[str,Any]] = field(default_factory=list)
    # Sidecar metadata (never part of the design). `group_mode` records which
    # block construction produced this analysis; `policies` are validated
    # GroupPolicy records (hierarchical mode); role_* describe role evidence.
    group_mode: str = "legacy"
    policies: list[Any] = field(default_factory=list)
    policy_status: dict[str, Any] = field(default_factory=dict)
    role_sources: dict[str, str] = field(default_factory=dict)
    role_confidence: dict[str, float] = field(default_factory=dict)
    role_status: dict[str, Any] = field(default_factory=dict)
    role_overrides: dict[str,str] = field(default_factory=dict)
    _source: Any = field(default=None, repr=False, compare=False)
    _graph_cache: Any = field(default=None, repr=False, compare=False)

    @property
    def graph(self):
        """AnalysisGraph sidecar (engine.relations), built lazily and cached."""
        if self._graph_cache is None and self._source is not None:
            from .relations import build_graph
            build_graph(self._source, self)
        return self._graph_cache

    def block(self, role: str) -> Block | None:
        return next((b for b in self.blocks if b.role == role), None)

    def all_blocks(self) -> list[Block]:
        return self.blocks + self.stickers + self.accents

    def to_dict(self) -> dict[str, Any]:
        return {
            "canvas": {"width": self.sw, "height": self.sh},
            "roles": self.roles,
            "blocks": [b.describe() for b in self.blocks],
            "stickers": [b.describe() for b in self.stickers],
            "accents": [b.describe() for b in self.accents],
            "background": [e["id"] for e in self.background],
            "decorations": [e["id"] for e in self.decorations],
            "notes": self.notes,
            "panels": self.panels,
            "relationships": self.relationships,
            # Additive key: existing consumers read the keys above only.
            "graph": self._graph_dict(),
        }

    def _graph_dict(self) -> dict[str, Any] | None:
        try:
            g = self.graph
            return g.to_dict() if g is not None else None
        except Exception as exc:  # the sidecar must never break analysis
            return {"version": None, "status": f"unavailable: {type(exc).__name__}"}


def _contains_center(outer: Box, inner: Box) -> bool:
    return outer.x <= inner.cx <= outer.x1 and outer.y <= inner.cy <= outer.y1


GROUP_MODES = ("legacy", "hierarchical")
DEFAULT_POLICY_FEATURES = frozenset({"graphics", "lockups", "stacks", "accents", "logos", "tables", "panels"})
POLICY_FEATURES = DEFAULT_POLICY_FEATURES | {"card_images", "tiles"}


def group_mode(value: str | None = None) -> str:
    """Explicit keyword first, then CHAMELEON_GROUP_POLICIES; default legacy."""
    v = (value if value is not None else os.environ.get("CHAMELEON_GROUP_POLICIES", "legacy")).strip().lower()
    return v if v in GROUP_MODES else "legacy"


def policy_features(value: str | None = None) -> frozenset:
    """Ablation control for hierarchical mode (CHAMELEON_GROUP_POLICY_FEATURES,
    comma list; default all)."""
    v = value if value is not None else os.environ.get("CHAMELEON_GROUP_POLICY_FEATURES", "")
    chosen = {x.strip() for x in v.split(",") if x.strip()}
    return frozenset(chosen & POLICY_FEATURES) if chosen else DEFAULT_POLICY_FEATURES


def apply_role_overrides(analysis,roles):
    """Apply validated human role intent without changing membership or source."""
    analysis.role_overrides.update(roles)
    analysis.roles.update(roles)
    for eid in roles:
        analysis.role_sources[eid]='explicit';analysis.role_confidence[eid]=1.0
    def visit(block):
        if len(block.ids)==1 and block.ids[0] in roles:
            role=roles[block.ids[0]];block.role=role
            block.label=LABELS.get(role,role);block.importance=IMPORTANCE.get(role,.5)
        if hasattr(block,'member_roles'):
            block.member_roles.update({eid:role for eid,role in roles.items() if eid in block.ids})
        for child in getattr(block,'children',()):visit(child)
    for block in analysis.all_blocks():visit(block)
    for attr in ('blocks','stickers','accents'):
        kept=[]
        for block in getattr(analysis,attr):
            if len(block.ids)==1 and roles.get(block.ids[0])=='decoration':
                member=block.members[0]
                if not any(e['id']==member['id'] for e in analysis.decorations):analysis.decorations.append(member)
            else:kept.append(block)
        setattr(analysis,attr,kept)
    analysis._graph_cache=None


def apply_group_mode_overrides(analysis,source,modes):
    """Rebuild supported layout blocks; reject unsupported structural promises."""
    from dataclasses import replace
    from .group_policies import MODES,policies_from_blocks,validate
    from .blocks import CardBlock,StrictRigidBlock,StackBlock
    if not isinstance(modes,dict):raise ValueError('group_modes must be a mapping')
    policies=analysis.policies or policies_from_blocks(source,analysis)
    by={policy.group_id:policy for policy in policies}
    blocks={}
    def inspect(block):
        blocks[block.id]=block
        for child in getattr(block,'children',()):inspect(child)
    for block in analysis.all_blocks():inspect(block)
    changes={}
    for gid,mode in modes.items():
        if gid not in by or gid not in blocks or mode not in MODES:raise ValueError('Unknown group or unsupported mode: '+str(gid))
        policy=by[gid];block=blocks[gid]
        allowed=policy.to_dict()['editable_modes']
        if mode not in allowed:raise ValueError('Mode cannot be applied safely to group '+gid)
        if mode=='reflowable' and mode!=policy.mode and (not getattr(block,'children',None) or hasattr(block,'panel')):
            raise ValueError('Only measured child-bearing groups may be made reflowable')
        if mode=='support' and not isinstance(block,CardBlock):raise ValueError('Support mode requires a recognized opaque support card')
        changes[gid]=replace(policy,mode=mode,axes=() if mode=='rigid' else policy.axes or ('y','x'),
                             authority='explicit',provenance='explicit:user group mode')
    def rebuild(block):
        if hasattr(block,'children'):
            block.children=[rebuild(child) for child in block.children]
        if block.id not in changes:return block
        policy=changes[block.id]
        if policy.mode=='rigid':
            return StrictRigidBlock(block.members,block.id,block.role,block.label,block.importance,policy)
        if policy.mode=='reflowable' and by[block.id].mode!='reflowable':
            return StackBlock(block.children,block.id,block.role,block.label,block.importance,
                              analysis.sw/analysis.sh,policy.axes,policy)
        block.policy=policy
        return block
    for attr in ('blocks','stickers','accents'):setattr(analysis,attr,[rebuild(block) for block in getattr(analysis,attr)])
    analysis.policies=validate([changes.get(policy.group_id,policy) for policy in policies],[e['id'] for e in source['elements']])
    analysis.policy_status={**analysis.policy_status,'user_overrides':dict(modes)}
    analysis._graph_cache=None


def analyze(source: dict[str, Any], *, group_policies: str | None = None, features: str | None = None,
            overrides: dict[str,Any] | None = None) -> Analysis:
    """Analyse `source`. `group_policies` ('legacy'|'hierarchical') overrides
    the CHAMELEON_GROUP_POLICIES switch. A failure of the hierarchical path
    falls back to legacy analysis with an explicit `policy_status`."""
    def finish(A):
        A._source=source
        if overrides is not None:
            if not isinstance(overrides,dict) or set(overrides)-{'roles','group_modes'} or not isinstance(overrides.get('roles',{}),dict) or not isinstance(overrides.get('group_modes',{}),dict):
                raise ValueError('Overrides require roles and/or group_modes mappings')
            roles=overrides.get('roles',{});by={e['id']:e for e in source['elements']}
            allowed={'text':{'headline','subhead','body','cta','badge','legal','logo','unknown'},
                     'image':{'hero','image','logo','cta','badge','decoration','unknown'},
                     'shape':{'cta','badge','unknown'}}
            for eid,role in roles.items():
                if eid not in by or not isinstance(role,str) or role not in allowed[by[eid]['type']]:
                    raise ValueError('Unknown element or unsupported type/role override: '+str(eid))
                if not any(eid in block.ids for block in A.all_blocks()):
                    raise ValueError('Background/decorative ownership cannot be promoted by a role-only override: '+str(eid))
            apply_role_overrides(A,roles)
            if overrides.get('group_modes'):apply_group_mode_overrides(A,source,overrides['group_modes'])
        return A
    mode = group_mode(group_policies)
    if mode == "hierarchical":
        try:
            A = _analyze_hierarchical(source, policy_features(features))
            return finish(A)
        except Exception as exc:  # typed fallback, never a crash
            A = _analyze_legacy(source)
            A.policy_status = {"status": "fallback_legacy", "error": f"{type(exc).__name__}: {exc}"[:300]}
            return finish(A)
    A = _analyze_legacy(source)
    return finish(A)


def _analyze_legacy(source: dict[str, Any]) -> Analysis:
    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    canvas = Box(0, 0, sw, sh)
    A = Analysis(sw, sh)
    els = list(source["elements"])
    min_z = min((e.get("z_index", 0) for e in els), default=0)

    # 1. Backgrounds: near-full-canvas layers at the bottom of the stack.
    rest = []
    for e in els:
        cover = element_box(e).intersection_area(canvas) / canvas.area
        if e["type"] in ("image", "shape") and cover >= 0.85 and e.get("z_index", 0) <= min_z + 1 and not e.get("group_id"):
            A.background.append(e)
            A.roles[e["id"]] = "background"
        else:
            rest.append(e)

    # 2. Explicit groups.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loose: list[dict[str, Any]] = []
    for e in rest:
        (groups[e["group_id"]] if e.get("group_id") else loose).append(e)

    # 3. Loose shapes: a shape hosting exactly one loose label is an inferred
    #    button; faint/bleeding shapes without text are decoration.
    loose_text = [e for e in loose if e["type"] == "text"]
    claimed: set[str] = set()
    inferred_buttons: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for e in [e for e in loose if e["type"] == "shape"]:
        eb = element_box(e)
        hosted = [t for t in loose_text if t["id"] not in claimed and _contains_center(eb, element_box(t))]
        if len(hosted) == 1 and eb.area < 0.3 * canvas.area:
            inferred_buttons.append((e, hosted[0]))
            claimed.update({e["id"], hosted[0]["id"]})
            continue
        A.decorations.append(e)
        A.roles[e["id"]] = "decoration"
        claimed.add(e["id"])
    loose = [e for e in loose if e["id"] not in claimed]

    # Large, uniform raster layers behave as panels, not hero photographs.
    # Keep their original assets and aspect; they are independently positioned.
    for e in list(loose):
        if e['type']=='image' and flat_asset(e.get('asset_id')) and element_box(e).area >= .08*canvas.area:
            A.decorations.append(e);A.roles[e['id']]='decoration';loose.remove(e)

    # 4. Groups -> blocks.
    buttons: list[ButtonBlock] = []
    for gid, members in groups.items():
        rotated = any(abs(float(m.get("rotation") or 0)) > 0.5 for m in members)
        shapes = [m for m in members if m["type"] == "shape"]
        texts = [m for m in members if m["type"] == "text"]
        if rotated:
            A.stickers.append(RigidBlock(members, gid, "sticker", LABELS["sticker"], IMPORTANCE["sticker"]))
        elif len(members) == 2 and len(shapes) == 1 and len(texts) == 1 and _contains_center(element_box(shapes[0]), element_box(texts[0])):
            buttons.append(ButtonBlock(shapes[0], texts[0], "badge", LABELS["badge"], IMPORTANCE["badge"]))
        else:
            A.blocks.append(RigidBlock(members, gid, "group", LABELS["group"], IMPORTANCE["group"]))
    for shape, text in inferred_buttons:
        buttons.append(ButtonBlock(shape, text, "badge", LABELS["badge"], IMPORTANCE["badge"]))

    # CTA: the largest button, preferring the lower half of the design.
    if buttons:
        cta = max(buttons, key=lambda b: b.src_box.area * (1.5 if b.src_box.cy > sh * 0.5 else 1.0))
        cta.role, cta.label, cta.importance = "cta", LABELS["cta"], IMPORTANCE["cta"]
        badges = [b for b in buttons if b is not cta]
        A.blocks.append(cta)
        # Badges sharing a baseline become one row.
        rows: list[list[ButtonBlock]] = []
        for b in sorted(badges, key=lambda b: b.src_box.cy):
            if rows and abs(rows[-1][0].src_box.cy - b.src_box.cy) < 0.5 * b.src_box.h:
                rows[-1].append(b)
            else:
                rows.append([b])
        for i, row in enumerate(rows):
            if len(row) > 1:
                A.blocks.append(RowBlock(row, f"badge_row_{i + 1}", "badges", LABELS["badges"], IMPORTANCE["badges"]))
            else:
                A.blocks.append(row[0])

    # 5. Loose images.
    images = [e for e in loose if e["type"] == "image"]
    def vis_area(e: dict[str, Any]) -> float:
        return visual_box(e).area / canvas.area

    content_imgs = [e for e in images if asset_exists(e.get("asset_id"))]
    for e in images:
        if not asset_exists(e.get("asset_id")):
            A.notes.append(f"Asset '{e.get('asset_id')}' for '{e['id']}' is missing - kept as a small placeholder accent.")
    hero = max((e for e in content_imgs if vis_area(e) >= 0.02), key=vis_area, default=None)
    logo = None
    others = [e for e in content_imgs if e is not hero]
    if others:
        def logo_score(e: dict[str, Any]) -> float:
            vb = visual_box(e)
            return (vb.w / max(1.0, vb.h) >= 1.3) * 2 + (vb.cy < sh * 0.35) * 2 - vis_area(e) * 5
        cand = max(others, key=logo_score)
        if logo_score(cand) >= 2:
            logo = cand
    for e in images:
        if e is hero:
            A.blocks.append(ImageBlock(e, "hero", LABELS["hero"], IMPORTANCE["hero"]))
        elif e is logo:
            A.blocks.append(ImageBlock(e, "logo", LABELS["logo"], IMPORTANCE["logo"]))
        elif e not in content_imgs or vis_area(e) < 0.03:
            A.accents.append(ImageBlock(e, "accent", LABELS["accent"], IMPORTANCE["accent"]))
        else:
            A.blocks.append(ImageBlock(e, "image", LABELS["image"], IMPORTANCE["image"]))

    # 6. Loose text: hierarchy from font size, fine print from size + position.
    texts = sorted([e for e in loose if e["type"] == "text"], key=lambda e: -float(e["font_size"]))
    top_fs = float(texts[0]["font_size"]) if texts else 0
    for i, e in enumerate(texts):
        fs = float(e["font_size"])
        if i == 0:
            role = "headline"
        elif fs <= min(16.0, max(10.0, 0.35 * top_fs)) and element_box(e).cy > sh * 0.75:
            role = "legal"
        elif i == 1:
            role = "subhead"
        else:
            role = "body"
        A.blocks.append(TextBlock(e, role, LABELS[role], IMPORTANCE[role]))

    # Overlap alone is a relationship, never evidence of a rigid group.
    # Explicit groups and small holders were handled above. In particular,
    # a panel containing paragraphs must not absorb those paragraphs.

    # Reading order: top-to-bottom bands, then left-to-right.
    band = 0.06 * sh
    A.blocks.sort(key=lambda b: (round(b.src_box.y / band), b.src_box.x))

    for b in A.all_blocks():
        for m in b.members:
            A.roles[m["id"]] = b.role
    # Record bounded containment and alignment relations, without joining every
    # overlapping object into a single component.
    for panel in A.decorations:
        if panel.get('opacity',1)<.5:continue
        if panel.get('type')=='image' and not flat_asset(panel.get('asset_id')):continue
        pb=element_box(panel)
        children=[e['id'] for e in source['elements'] if e['type']=='text' and e.get('z_index',0)>panel.get('z_index',0) and pb.intersection_area(element_box(e))>=.8*max(1,element_box(e).area)]
        if children:A.panels.append({'id':panel['id'],'children':children,'box':pb.to_dict()})
    for i,a in enumerate(A.all_blocks()):
        for b in A.all_blocks()[i+1:]:
            x,y=a.src_box,b.src_box
            if abs(x.x-y.x)<.015*sw:A.relationships.append({'kind':'left_alignment','a':a.id,'b':b.id,'confidence':.8})
            if x.intersection_area(y)>0:A.relationships.append({'kind':'intentional_overlap','a':a.id,'b':b.id,'confidence':.7})
    return A


# ---------------------------------------------------------------------------
# Hierarchical analysis (CHAMELEON_GROUP_POLICIES=hierarchical)
# ---------------------------------------------------------------------------


def _frame_anchored(e: dict[str, Any], sw: float, sh: float) -> bool:
    """Touches or bleeds past at least two canvas sides (a frame graphic)."""
    b = element_box(e)
    return sum((b.x <= 1, b.y <= 1, b.x1 >= sw - 1, b.y1 >= sh - 1)) >= 2


def _text_roles(texts: list[dict[str, Any]], sh: float) -> dict[str, str]:
    """Legacy text hierarchy (font-size rank, fine print by size + position)."""
    texts = sorted(texts, key=lambda e: -float(e["font_size"]))
    top_fs = float(texts[0]["font_size"]) if texts else 0
    roles = {}
    for i, e in enumerate(texts):
        fs = float(e["font_size"])
        if i == 0:
            role = "headline"
        elif fs <= min(16.0, max(10.0, 0.35 * top_fs)) and element_box(e).cy > sh * 0.75:
            role = "legal"
        elif i == 1:
            role = "subhead"
        else:
            role = "body"
        roles[e["id"]] = role
    return roles


_RANK = ["headline", "logo", "cta", "subhead", "hero", "image", "badges", "badge", "body", "legal", "accent"]


def _analyze_hierarchical(source: dict[str, Any], features: frozenset = DEFAULT_POLICY_FEATURES) -> Analysis:
    from .group_policies import InferenceContext, infer_lockups, policies_from_blocks, validate
    from .relations import alpha_coverage, asset_stats, glyph_box
    from .blocks import CardBlock, GridBlock, LockupBlock, StackBlock

    sw, sh = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    canvas = Box(0, 0, sw, sh)
    A = Analysis(sw, sh)
    A.group_mode = "hierarchical"
    els = list(source["elements"])
    min_z = min((e.get("z_index", 0) for e in els), default=0)
    kinds = {e["id"]: asset_stats(e.get("asset_id")).kind for e in els if e["type"] == "image"}

    # 1. Backgrounds (unchanged rule).
    rest = []
    for e in els:
        cover = element_box(e).intersection_area(canvas) / canvas.area
        if e["type"] in ("image", "shape") and cover >= 0.85 and e.get("z_index", 0) <= min_z + 1 and not e.get("group_id"):
            A.background.append(e)
            A.roles[e["id"]] = "background"
        else:
            rest.append(e)

    # 2. Explicit groups (highest authority) vs loose elements.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loose: list[dict[str, Any]] = []
    for e in rest:
        (groups[e["group_id"]] if e.get("group_id") else loose).append(e)

    # 3. Loose shapes: single hosted label -> inferred button; otherwise decoration.
    loose_text = [e for e in loose if e["type"] == "text"]
    claimed: set[str] = set()
    inferred_buttons: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for e in [e for e in loose if e["type"] == "shape"]:
        eb = element_box(e)
        hosted = [t for t in loose_text if t["id"] not in claimed and _contains_center(eb, element_box(t))]
        if len(hosted) == 1 and eb.area < 0.3 * canvas.area:
            inferred_buttons.append((e, hosted[0]))
            claimed.update({e["id"], hosted[0]["id"]})
            continue
        A.decorations.append(e)
        A.roles[e["id"]] = "decoration"
        claimed.add(e["id"])
    loose = [e for e in loose if e["id"] not in claimed]

    # Large uniform rasters are panels (legacy rule). Large frame-anchored
    # graphics (few flat colours, few internal edges, touching two canvas
    # sides) are ornamental backplates, not hero photographs.
    for e in list(loose):
        if e["type"] != "image":
            continue
        area = element_box(e).area
        if flat_asset(e.get("asset_id")) and area >= .08 * canvas.area:
            A.decorations.append(e); A.roles[e["id"]] = "decoration"; loose.remove(e)
        elif "graphics" in features and kinds.get(e["id"]) in ("graphic", "flat", "linework") and visual_box(e).area >= .08 * canvas.area \
                and _frame_anchored(e, sw, sh):
            A.decorations.append(e); A.roles[e["id"]] = "decoration"; loose.remove(e)
            A.role_sources[e["id"]] = "heuristic"

    # 4. Explicit groups -> blocks (legacy semantics).
    buttons: list[ButtonBlock] = []
    for gid, members in groups.items():
        rotated = any(abs(float(m.get("rotation") or 0)) > 0.5 for m in members)
        shapes = [m for m in members if m["type"] == "shape"]
        texts = [m for m in members if m["type"] == "text"]
        if rotated:
            A.stickers.append(RigidBlock(members, gid, "sticker", LABELS["sticker"], IMPORTANCE["sticker"]))
        elif len(members) == 2 and len(shapes) == 1 and len(texts) == 1 and _contains_center(element_box(shapes[0]), element_box(texts[0])):
            buttons.append(ButtonBlock(shapes[0], texts[0], "badge", LABELS["badge"], IMPORTANCE["badge"]))
        else:
            A.blocks.append(RigidBlock(members, gid, "group", LABELS["group"], IMPORTANCE["group"]))
    for shape, text in inferred_buttons:
        buttons.append(ButtonBlock(shape, text, "badge", LABELS["badge"], IMPORTANCE["badge"]))
    if buttons:
        cta = max(buttons, key=lambda b: b.src_box.area * (1.5 if b.src_box.cy > sh * 0.5 else 1.0))
        cta.role, cta.label, cta.importance = "cta", LABELS["cta"], IMPORTANCE["cta"]
        A.blocks.append(cta)
        rows: list[list[ButtonBlock]] = []
        for b in sorted([b for b in buttons if b is not cta], key=lambda b: b.src_box.cy):
            if rows and abs(rows[-1][0].src_box.cy - b.src_box.cy) < 0.5 * b.src_box.h:
                rows[-1].append(b)
            else:
                rows.append([b])
        for i, row in enumerate(rows):
            A.blocks.append(RowBlock(row, f"badge_row_{i + 1}", "badges", LABELS["badges"], IMPORTANCE["badges"]) if len(row) > 1 else row[0])

    # 5. Provisional image roles (legacy rules) on the remaining loose images.
    images = [e for e in loose if e["type"] == "image"]

    def vis_area(e: dict[str, Any]) -> float:
        return visual_box(e).area / canvas.area

    content_imgs = [e for e in images if asset_exists(e.get("asset_id"))]
    for e in images:
        if not asset_exists(e.get("asset_id")):
            A.notes.append(f"Asset '{e.get('asset_id')}' for '{e['id']}' is missing - kept as a small placeholder accent.")
    # Sparse line art (rules, table ruling, outlines) is never the hero subject.
    hero_ok = [e for e in content_imgs if "graphics" not in features or kinds.get(e["id"]) != "linework"]
    hero = max((e for e in hero_ok if vis_area(e) >= 0.02), key=vis_area, default=None)
    logo = None
    others = [e for e in content_imgs if e is not hero]
    if others:
        def logo_score(e: dict[str, Any]) -> float:
            vb = visual_box(e)
            return (vb.w / max(1.0, vb.h) >= 1.3) * 2 + (vb.cy < sh * 0.35) * 2 - vis_area(e) * 5
        cand = max(others, key=logo_score)
        if logo_score(cand) >= 2:
            logo = cand
    img_role: dict[str, str] = {}
    for e in images:
        if e is hero:
            img_role[e["id"]] = "hero"
        elif e is logo:
            img_role[e["id"]] = "logo"
        elif e not in content_imgs or vis_area(e) < 0.03:
            img_role[e["id"]] = "accent"
        else:
            img_role[e["id"]] = "image"
    texts = [e for e in loose if e["type"] == "text"]
    text_role = _text_roles(texts, sh)

    # 6. Inferred lockups / stacks / tables among loose elements.
    glyph = {t["id"]: glyph_box(t) for t in texts}
    ctx = InferenceContext(source, glyph, kinds, img_role, texts,
                           [e for e in images if asset_exists(e.get("asset_id")) and e is not hero], features & {"lockups", "stacks", "accents", "logos", "tables"})
    inferred = infer_lockups(ctx)
    by_id = {e["id"]: e for e in els}
    by_pol = {p.group_id: p for p in inferred}

    def unit_role(ids) -> str:
        if any(img_role.get(i) == "logo" for i in ids):
            return "logo"
        rs = [text_role[i] for i in ids if i in text_role] or [img_role.get(i, "image") for i in ids]
        return min(rs, key=lambda r: _RANK.index(r) if r in _RANK else len(_RANK))

    def make_block(p):
        if p.kind == "display_stack":
            kids = []
            for u in p.source_order:
                if u in by_pol:
                    q = by_pol[u]
                    kids.append(LockupBlock([by_id[i] for i in q.member_ids], q.group_id, unit_role(q.member_ids), "Lockup", IMPORTANCE.get(unit_role(q.member_ids), .5), q, text_role))
                else:
                    r = text_role.get(u, "body")
                    kids.append(TextBlock(by_id[u], r, LABELS[r], IMPORTANCE[r]))
            r = unit_role(p.member_ids)
            return StackBlock(kids, p.group_id, r, LABELS.get(r, "Group"), IMPORTANCE.get(r, .5), sw / sh, p.axes, p)
        r = unit_role(p.member_ids)
        return LockupBlock([by_id[i] for i in p.member_ids], p.group_id, r, LABELS.get(r, "Group"), IMPORTANCE.get(r, .5), p, text_role)

    in_policy: set[str] = set()
    for p in inferred:
        if p.parent is None:
            A.blocks.append(make_block(p))
            in_policy.update(p.member_ids)

    # 7. Remaining loose images and texts as single-element blocks.
    for e in images:
        if e["id"] in in_policy:
            continue
        r = img_role[e["id"]]
        (A.accents if r == "accent" else A.blocks).append(ImageBlock(e, r, LABELS[r], IMPORTANCE[r]))
    for e in sorted([t for t in texts if t["id"] not in in_policy], key=lambda e: -float(e["font_size"])):
        r = text_role[e["id"]]
        if abs(float(e.get("rotation") or 0)) > 0.5:
            # Rotated copy is laid as a rigid unit through its rotated extent;
            # a horizontal text measure would misjudge its footprint.
            from .group_policies import GroupPolicy
            pol = GroupPolicy(f"rotated:{e['id']}", (e["id"],), None, "heuristic", "rigid", (), (e["id"],), "source", None,
                              "heuristic:rotated text measured by its rotated extent", kind="rotated_text", confidence=0.9)
            A.blocks.append(LockupBlock([e], pol.group_id, r, LABELS[r], IMPORTANCE[r], pol, text_role))
        else:
            A.blocks.append(TextBlock(e, r, LABELS[r], IMPORTANCE[r]))

    # Support cards: a flat panel (uniform raster or shape) whose opaque area
    # carries copy becomes one unit with that copy; the panel follows it.
    if "panels" in features:
        from .blocks import CardBlock
        for panel in sorted(list(A.decorations), key=lambda e: e.get("z_index", 0), reverse=True):
            if panel.get("opacity", 1) < .5 or abs(float(panel.get("rotation") or 0)) > 0.5:
                continue
            if panel["type"] == "image" and not (flat_asset(panel.get("asset_id")) or kinds.get(panel["id"]) == "flat"):
                continue
            pb = element_box(panel)
            if pb.intersection_area(canvas) > .8 * canvas.area:
                continue
            kids = []
            for b in A.blocks:
                if isinstance(b, (ImageBlock, CardBlock)) or not any(m["type"] == "text" for m in b.members):
                    continue
                if any(m.get("z_index", 0) <= panel.get("z_index", 0) for m in b.members):
                    continue
                ok = True
                for m in b.members:
                    vb = glyph_box(m) if m["type"] == "text" else visual_box(m)
                    if pb.intersection_area(vb) < .8 * max(1e-6, vb.area) or (m["type"] == "text" and alpha_coverage(panel, vb) < .9):
                        ok = False
                        break
                if ok:
                    kids.append(b)
            if not kids:
                continue
            if "card_images" in features:
                # An illustration painted on the card's opaque area (most of its
                # visible pixels over the panel) travels with the card.
                for b in A.blocks+A.accents:
                    if not isinstance(b, ImageBlock) or b.members[0].get("z_index", 0) <= panel.get("z_index", 0):
                        continue
                    vb = visual_box(b.members[0])
                    if vb.area > 0 and pb.intersection_area(vb) >= .8 * vb.area and alpha_coverage(panel, vb) >= .8:
                        kids.append(b)
            role = min((k.role for k in kids), key=lambda r: _RANK.index(r) if r in _RANK else len(_RANK))
            from .group_policies import GroupPolicy
            pol = GroupPolicy(f"card:{panel['id']}", tuple([panel["id"]] + [i for k in kids for i in k.ids]), None, "heuristic", "support",
                              ("y", "x"), tuple(k.id for k in kids), "source", None,
                              "heuristic:flat panel whose opaque area carries the copy", kind="card", support_id=panel["id"], confidence=0.6)
            card = CardBlock(panel, kids, pol.group_id, role, LABELS.get(role, "Group"), IMPORTANCE.get(role, .5), sw / sh, pol)
            A.blocks = [b for b in A.blocks if b not in kids] + [card]
            A.accents = [b for b in A.accents if b not in kids]
            A.decorations.remove(panel)
            A.roles[panel["id"]] = "decoration"

        # Cards whose panels tile the canvas (abutting backplates) form one
        # reflowable grid: source arrangement, a row on a much wider canvas,
        # a column on a much taller one. Members keep their own card layout.
        cards = [b for b in A.blocks if isinstance(b, CardBlock)]
        if "tiles" in features and len(cards) >= 2:
            from .blocks import GridBlock
            from .group_policies import GroupPolicy, _UF
            uf = _UF([c.id for c in cards])
            tol = .012 * min(sw, sh)
            for i, a in enumerate(cards):
                for b in cards[i + 1:]:
                    pa, pb2 = element_box(a.panel), element_box(b.panel)
                    gx = max(pa.x, pb2.x) - min(pa.x1, pb2.x1)
                    gy = max(pa.y, pb2.y) - min(pa.y1, pb2.y1)
                    # Edge-adjacent: touching along one axis, overlapping along the other.
                    if (abs(gx) <= tol and gy < -tol) or (abs(gy) <= tol and gx < -tol):
                        uf.union(a.id, b.id)
            by_card = {c.id: c for c in cards}
            for n, g in enumerate(uf.groups()):
                if len(g) < 2:
                    continue
                kids = [by_card[i] for i in g]
                role = min((k.role for k in kids), key=lambda r: _RANK.index(r) if r in _RANK else len(_RANK))
                gid = f"tiles_{n + 1}"
                kids = [CardBlock(k.panel, k.children, k.id, k.role, k.label, k.importance, k.src_aspect,
                                  GroupPolicy(k.policy.group_id, k.policy.member_ids, gid, k.policy.authority, k.policy.mode, k.policy.axes,
                                              k.policy.source_order, k.policy.alignment, k.policy.padding_bounds, k.policy.provenance,
                                              k.policy.kind, k.policy.children, k.policy.support_id, k.policy.confidence)) for k in kids]
                pol = GroupPolicy(gid, tuple(i for k in kids for i in k.ids), None, "heuristic", "reflowable", ("source", "x", "y"),
                                  tuple(k.id for k in kids), "source", None, "heuristic:abutting support panels tile the canvas",
                                  kind="card_grid", children=tuple(k.id for k in kids), confidence=0.55)
                grid = GridBlock(kids, gid, role, LABELS.get(role, "Group"), IMPORTANCE.get(role, .5), sw / sh, pol)
                A.blocks = [b for b in A.blocks if b.id not in g] + [grid]

    band = 0.06 * sh
    A.blocks.sort(key=lambda b: (round(b.src_box.y / band), b.src_box.x))
    for b in A.all_blocks():
        for m in b.members:
            # Inside inferred lockups every element keeps its own role evidence.
            if isinstance(b, (LockupBlock, StackBlock, CardBlock, GridBlock)):
                # Every element keeps its own evidence; images inside a
                # non-logo lockup are its accents (rules, icons, ruling).
                panels = {c.panel["id"] for c in getattr(b,"children",[]) if isinstance(c,CardBlock)} | ({b.panel["id"]} if isinstance(b,CardBlock) else set())
                r = text_role.get(m["id"]) or img_role.get(m["id"]) or ("decoration" if m["id"] in panels else b.role)
                keep = r in ("logo","decoration") or (r in ("hero","image") and isinstance(b,(CardBlock,GridBlock)))
                A.roles[m["id"]] = r if m["type"] != "image" or keep else "accent"
            else:
                A.roles[m["id"]] = b.role
    for e in els:
        A.role_sources.setdefault(e["id"], "explicit" if e.get("group_id") and e["id"] in A.roles and A.roles[e["id"]] in ("badge", "badges", "cta", "group", "sticker") else "heuristic")

    # 8. Support panels: flat decorations whose *opaque* area carries copy.
    owner = {i: b for b in A.all_blocks() for i in b.ids}
    for panel in A.decorations:
        if panel.get('opacity', 1) < .5:
            continue
        if panel.get('type') == 'image' and not flat_asset(panel.get('asset_id')):
            continue
        pb = element_box(panel)
        kids = []
        for e in source['elements']:
            if e['type'] != 'text' or e.get('z_index', 0) <= panel.get('z_index', 0):
                continue
            tb = element_box(e)
            if pb.intersection_area(tb) < .8 * max(1, tb.area):
                continue
            if "panels" in features and alpha_coverage(panel, glyph_box(e)) < .9:
                continue  # inside the box but on a transparent part of the asset
            kids.append(e['id'])
        if "panels" in features:
            # The whole inferred unit of a contained text follows the panel (accent + copy).
            for t in list(kids):
                b = owner.get(t)
                if isinstance(b, LockupBlock) and b.kind in ("accent_lockup", "text_lockup", "logo_lockup"):
                    for i in b.ids:
                        if i not in kids and pb.intersection_area(visual_box(by_id[i])) >= .8 * max(1, visual_box(by_id[i]).area):
                            kids.append(i)
        if kids:
            A.panels.append({'id': panel['id'], 'children': kids, 'box': pb.to_dict()})
    for i, a in enumerate(A.all_blocks()):
        for b in A.all_blocks()[i + 1:]:
            x, y = a.src_box, b.src_box
            if abs(x.x - y.x) < .015 * sw:
                A.relationships.append({'kind': 'left_alignment', 'a': a.id, 'b': b.id, 'confidence': .8})
            if x.intersection_area(y) > 0:
                A.relationships.append({'kind': 'intentional_overlap', 'a': a.id, 'b': b.id, 'confidence': .7})

    # 9. Validated policy hierarchy (explicit, legacy-compatible, inferred).
    try:
        pols = policies_from_blocks(source, A)
        names = {p.group_id for p in pols}
        extra = [p for p in inferred if p.group_id not in names]
        A.policies = validate(pols + extra, [e["id"] for e in els])
        A.policy_status = {"status": "ok", "inferred": len(inferred), "features": sorted(features)}
    except Exception as exc:
        A.policies = []
        A.policy_status = {"status": "policies_invalid", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return A
