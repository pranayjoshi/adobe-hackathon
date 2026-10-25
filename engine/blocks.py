"""Layout blocks: the units an arrangement moves around.

Every block answers one question - "how big are you at scale `s` when you may
use at most `max_w` pixels of width?" - and returns a `Laid` whose `place()`
emits geometry patches keyed by original element IDs. Blocks never touch
content or style; they only produce x/y/width/height/font_size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .canonical import Box, element_box, rotated_extent, union_all, visual_fraction
from .typography import TextFit, measure, source_lines, tight_width

Patches = dict[str, dict[str, float]]

# Minimum comfortable font sizes (px) by role. Treated as preferences: the
# solver drops them when content cannot otherwise fit and reports an issue.
FLOORS = {
    "headline": 14,
    "subhead": 10,
    "body": 9,
    "legal": 7,
    "cta": 10,
    "badge": 8,
    "sticker": 8,
    "group": 8,
}


@dataclass
class Ctx:
    cw: float
    ch: float
    floors: bool = True
    boost: dict[str, float] = field(default_factory=dict)  # role -> scale multiplier

    def floor(self, role: str) -> int:
        return FLOORS.get(role, 8) if self.floors else 1


@dataclass
class Laid:
    w: float
    h: float
    place: Callable[[float, float], tuple[Patches, list[tuple[str, Box]]]]
    pad: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # transparent margin l/t/r/b


class Block:
    id: str
    role: str
    label: str
    importance: float
    members: list[dict[str, Any]]

    @property
    def src_box(self) -> Box:
        raise NotImplementedError

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        raise NotImplementedError

    @property
    def ids(self) -> list[str]:
        return [m["id"] for m in self.members]

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "label": self.label,
            "kind": type(self).__name__.replace("Block", "").lower(),
            "members": self.ids,
            "box": self.src_box.to_dict(),
            "importance": self.importance,
        }


def _shrink_for_width(el: dict[str, Any], fs: int, width: float) -> TextFit:
    fit = measure(el, fs, width)
    while fit.overflow and fit.font_size > 1:
        fit = measure(el, fit.font_size - 1, width)
    return fit


class TextBlock(Block):
    def __init__(self, el: dict[str, Any], role: str, label: str, importance: float):
        self.el, self.role, self.label, self.importance = el, role, label, importance
        self.id, self.members = el["id"], [el]
        self.src_fs = float(el["font_size"])
        self.src_lines = source_lines(el)

    @property
    def src_box(self) -> Box:
        return element_box(self.el)

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        s *= ctx.boost.get(self.role, 1.0)
        fs = max(1, round(self.src_fs * s), min(ctx.floor(self.role), int(self.src_fs)))
        fit = _shrink_for_width(self.el, fs, max_w)
        # Upscaling should not push a line-count increase: prefer the source wrap.
        while fit.font_size > self.src_fs and fit.lines > self.src_lines:
            fit = measure(self.el, fit.font_size - 1, max_w)
        width = tight_width(self.el, fit)
        el_id = self.el["id"]

        def place(x: float, y: float):
            patch = {"x": x, "y": y, "width": width, "height": fit.height, "font_size": fit.font_size}
            return {el_id: patch}, [(self.id, Box(x, y, width, fit.height))]

        return Laid(width, fit.height, place)


class ImageBlock(Block):
    def __init__(self, el: dict[str, Any], role: str, label: str, importance: float):
        self.el, self.role, self.label, self.importance = el, role, label, importance
        self.id, self.members = el["id"], [el]
        self.frac = visual_fraction(el)

    @property
    def src_box(self) -> Box:
        b = element_box(self.el)
        fl, ft, fw, fh = self.frac
        return Box(b.x + fl * b.w, b.y + ft * b.h, fw * b.w, fh * b.h)

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        s *= ctx.boost.get(self.role, 1.0)
        sw, sh = float(self.el["width"]), float(self.el["height"])
        fl, ft, fw, fh = self.frac
        s = min(s, max_w / max(1e-6, sw * fw), ctx.cw / sw, ctx.ch / sh)
        w, h = sw * s, sh * s
        vw, vh = w * fw, h * fh
        el_id = self.el["id"]

        def place(x: float, y: float):
            ex = min(max(0.0, x - fl * w), ctx.cw - w)
            ey = min(max(0.0, y - ft * h), ctx.ch - h)
            patch = {"x": ex, "y": ey, "width": w, "height": h}
            return {el_id: patch}, [(self.id, Box(ex + fl * w, ey + ft * h, vw, vh))]

        return Laid(vw, vh, place, (fl * w, ft * h, (1 - fl - fw) * w, (1 - ft - fh) * h))


class ButtonBlock(Block):
    """A shape with a label on it. Flexible: the label keeps a readable size
    and the shape is re-padded around it, rather than scaling as a bitmap."""

    def __init__(self, shape: dict[str, Any], text: dict[str, Any], role: str, label: str, importance: float):
        self.shape, self.text, self.role, self.label, self.importance = shape, text, role, label, importance
        self.id = shape.get("group_id") or f"{shape['id']}+{text['id']}"
        self.members = [shape, text]
        self.src_fs = float(text["font_size"])
        src_fit = measure(text, int(self.src_fs), float(text["width"]))
        self.src_lines = src_fit.lines
        self.pad_x = max(self.src_fs * 1.2, float(shape["width"]) - src_fit.line_width)
        self.pad_y = max(self.src_fs * 0.6, float(shape["height"]) - src_fit.height)

    @property
    def src_box(self) -> Box:
        return element_box(self.shape).union(element_box(self.text))

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        s *= ctx.boost.get(self.role, 1.0)
        fs = max(1, round(self.src_fs * s), min(ctx.floor(self.role), int(self.src_fs)))
        while True:
            min_pad = fs * 1.2
            pad_x = max(min_pad, self.pad_x * s)
            fit = measure(self.text, fs, 1e6)
            if fit.line_width + pad_x > max_w:
                pad_x = max(min_pad, max_w - fit.line_width)
            if fit.line_width + pad_x <= max_w or fs <= 1:
                break
            # Allow the label to wrap before shrinking it further.
            wrapped = measure(self.text, fs, max(1.0, max_w - min_pad))
            if not wrapped.overflow and wrapped.lines <= max(2, self.src_lines):
                fit, pad_x = wrapped, min_pad
                break
            fs -= 1
        pad_y = max(fs * 0.6, self.pad_y * s)
        tw = tight_width(self.text, fit)
        w = min(max_w, fit.line_width + pad_x)
        h = fit.height + pad_y
        sid, tid = self.shape["id"], self.text["id"]

        def place(x: float, y: float):
            label_h = 2 * fit.visual_center
            patches = {
                sid: {"x": x, "y": y, "width": w, "height": h},
                tid: {
                    "x": x + (w - tw) / 2,
                    "y": y + h / 2 - fit.visual_center,
                    "width": tw,
                    "height": label_h,
                    "font_size": fit.font_size,
                },
            }
            return patches, [(self.id, Box(x, y, w, h))]

        return Laid(w, h, place)


class RigidBlock(Block):
    """Members scale uniformly about the group origin (rotated stickers,
    text-on-image composites). Labels are re-centred on their shapes."""

    def __init__(self, members: list[dict[str, Any]], block_id: str, role: str, label: str, importance: float):
        self.members, self.id, self.role, self.label, self.importance = members, block_id, role, label, importance
        self.U = union_all(rotated_extent(m) for m in members)
        shapes = [m for m in members if m["type"] == "shape"]
        self.host: dict[str, dict[str, Any]] = {}
        for m in members:
            if m["type"] != "text":
                continue
            tb = element_box(m)
            for sh in shapes:
                sb = element_box(sh)
                if sb.x <= tb.cx <= sb.x1 and sb.y <= tb.cy <= sb.y1:
                    self.host[m["id"]] = sh
                    break
        self.src_lines = {m["id"]: source_lines(m) for m in members if m["type"] == "text"}

    @property
    def src_box(self) -> Box:
        return self.U

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        s *= ctx.boost.get(self.role, 1.0)
        U = self.U
        s = min(s, max_w / U.w, ctx.cw / U.w, ctx.ch / U.h)
        fits: dict[str, TextFit] = {}
        for m in self.members:
            if m["type"] != "text":
                continue
            width = float(m["width"]) * s
            fit = _shrink_for_width(m, max(1, round(float(m["font_size"]) * s)), width)
            while fit.lines > self.src_lines[m["id"]] and fit.font_size > 1:
                fit = measure(m, fit.font_size - 1, width)
            fits[m["id"]] = fit

        def place(x: float, y: float):
            patches: Patches = {}
            for m in self.members:
                b = element_box(m)
                cx, cy = x + (b.cx - U.x) * s, y + (b.cy - U.y) * s
                w, h = b.w * s, b.h * s
                patch = {"x": cx - w / 2, "y": cy - h / 2, "width": w, "height": h}
                if m["id"] in fits:
                    f = fits[m["id"]]
                    patch["font_size"] = f.font_size
                    if m["id"] in self.host:
                        patch["height"] = 2 * f.visual_center
                        patch["y"] = cy - f.visual_center
                patches[m["id"]] = patch
            return patches, [(self.id, Box(x, y, U.w * s, U.h * s))]

        return Laid(U.w * s, U.h * s, place)


class RowBlock(Block):
    """Repeated siblings (e.g. a row of badges). Wraps onto extra lines when
    the available width is too narrow, keeping the badges themselves intact."""

    def __init__(self, children: list[Block], block_id: str, role: str, label: str, importance: float):
        self.children = sorted(children, key=lambda b: b.src_box.x)
        self.id, self.role, self.label, self.importance = block_id, role, label, importance
        self.members = [m for c in self.children for m in c.members]
        gaps = [b.src_box.x - a.src_box.x1 for a, b in zip(self.children, self.children[1:])]
        self.gap = max(8.0, sum(gaps) / len(gaps)) if gaps else 16.0

    @property
    def src_box(self) -> Box:
        return union_all(c.src_box for c in self.children)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["children"] = [c.describe() for c in self.children]
        return d

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        gap = max(4.0, self.gap * s)
        laid = [c.layout(s, max_w, ctx) for c in self.children]
        lines: list[list[Laid]] = [[]]
        width = 0.0
        for item in laid:
            needed = item.w if not lines[-1] else width + gap + item.w
            if lines[-1] and needed > max_w:
                lines.append([item])
                width = item.w
            else:
                lines[-1].append(item)
                width = needed
        line_w = [sum(i.w for i in ln) + gap * (len(ln) - 1) for ln in lines]
        line_h = [max(i.h for i in ln) for ln in lines]
        W = max(line_w)
        H = sum(line_h) + gap * 0.6 * (len(lines) - 1)

        def place(x: float, y: float):
            patches: Patches = {}
            boxes: list[tuple[str, Box]] = []
            cy = y
            for ln, lw, lh in zip(lines, line_w, line_h):
                cx = x + (W - lw) / 2
                for item in ln:
                    p, b = item.place(cx, cy + (lh - item.h) / 2)
                    patches.update(p)
                    boxes.extend(b)
                    cx += item.w + gap
                cy += lh + gap * 0.6
            return patches, boxes + [(self.id, Box(x, y, W, H))]

        return Laid(W, H, place)


# ---------------------------------------------------------------------------
# Policy-driven blocks (CHAMELEON_GROUP_POLICIES=hierarchical). Legacy block
# construction never creates these, so legacy output is unchanged.
# ---------------------------------------------------------------------------


class LockupBlock(RigidBlock):
    """An inferred rigid unit (overlapping display type, an accent with the copy
    it annotates, a logo with its wordmark, a ruled table). Members keep their
    relative geometry up to uniform scale; `policy` records why."""

    def __init__(self, members, block_id, role, label, importance, policy=None, member_roles=None):
        super().__init__(members, block_id, role, label, importance)
        self.policy = policy
        self.kind = getattr(policy, "kind", "lockup")
        self.member_roles = dict(member_roles or {})
        # Source overflow ratio per text (imported boxes are often shorter
        # than the rendered line box); never make it worse.
        self.src_ratio = {}
        for m in members:
            if m["type"] == "text":
                f = measure(m, int(float(m["font_size"])), float(m["width"]))
                self.src_ratio[m["id"]] = max(1.0, f.height / max(1e-6, float(m["height"])))
        # Smallest uniform scale at which every member still renders inside its
        # scaled box (fixed letter-spacing and the 1px type floor stop text
        # shrinking proportionally). Below it the unit would come apart.
        self.s_min = 0.0
        for m in members:
            if m["type"] != "text" or not str(m.get("content", "")).strip():
                continue
            fs, bw = float(m["font_size"]), float(m["width"])
            lo = 0.0
            for k in (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45, 0.65, 1.0):
                f = measure(m, max(1, int(fs * k + 1e-6)), bw * k)
                if not f.overflow and f.font_size >= 2:
                    break
                lo = k
            self.s_min = max(self.s_min, lo)

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        s *= ctx.boost.get(self.role, 1.0)
        U = self.U
        # Like a TextBlock, a lockup does not shrink its copy below the role
        # floors (relaxed by the solver via ctx.floors=False); width is hard.
        for m in self.members:
            if m["type"] == "text" and str(m.get("content", "")).strip():
                fs = float(m["font_size"])
                s = max(s, min(ctx.floor(self.member_roles.get(m["id"], self.role)), fs) / fs)
        s = max(min(s, max_w / U.w, ctx.cw / U.w, ctx.ch / U.h), min(self.s_min, ctx.cw / U.w, ctx.ch / U.h))
        fits: dict[str, TextFit] = {}
        for m in self.members:
            if m["type"] != "text":
                continue
            width, height = float(m["width"]) * s, float(m["height"]) * s
            fit = measure(m, max(1, int(float(m["font_size"]) * s + 1e-6)), width)
            while fit.font_size > 1 and (fit.overflow or fit.lines > self.src_lines[m["id"]]
                                         or (m["id"] not in self.host and fit.height > self.src_ratio[m["id"]] * height + 0.01 * height)):
                fit = measure(m, fit.font_size - 1, width)
            fits[m["id"]] = fit

        def place(x: float, y: float):
            patches: Patches = {}
            for m in self.members:
                b = element_box(m)
                cx, cy = x + (b.cx - U.x) * s, y + (b.cy - U.y) * s
                w, h = b.w * s, b.h * s
                patch = {"x": cx - w / 2, "y": cy - h / 2, "width": w, "height": h}
                if m["id"] in fits:
                    f = fits[m["id"]]
                    patch["font_size"] = f.font_size
                    if m["id"] in self.host:
                        patch["height"] = 2 * f.visual_center
                        patch["y"] = cy - f.visual_center
                    else:
                        # At extreme reductions the 1px floor of type can exceed the
                        # scaled box: grow the box about its centre (relative
                        # position kept) rather than emit clipped text.
                        need_h = max(h, f.height / self.src_ratio[m["id"]] + 0.05)
                        need_w = max(w, f.line_width + 1.0) if f.overflow else w
                        patch.update(width=need_w, height=need_h, x=cx - need_w / 2, y=cy - need_h / 2)
                patches[m["id"]] = patch
            return patches, [(self.id, Box(x, y, U.w * s, U.h * s))]

        return Laid(U.w * s, U.h * s, place)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["kind"] = self.kind
        if self.policy is not None:
            d["policy"] = self.policy.group_id
        return d


class StrictRigidBlock(RigidBlock):
    """Explicit human rigid policy: one uniform geometry transform for leaves.

    Text chooses an integer fitting size; it never grows or recentres a child
    box as a side effect of the parent's transform.
    """
    def __init__(self,members,block_id,role,label,importance,policy=None):
        super().__init__(members,block_id,role,label,importance)
        self.policy=policy;self.kind=getattr(policy,'kind','explicit')

    def layout(self,s,max_w,ctx):
        from .group_policies import _rigid_patches
        s=max(1e-8,min(s*ctx.boost.get(self.role,1),max_w/self.U.w,ctx.cw/self.U.w,ctx.ch/self.U.h))
        patches,_,_=_rigid_patches(self.members,s)
        def place(x,y):
            return {eid:{**patch,'x':patch['x']+x,'y':patch['y']+y} for eid,patch in patches.items()},[(self.id,Box(x,y,self.U.w*s,self.U.h*s))]
        return Laid(self.U.w*s,self.U.h*s,place)


class StackBlock(Block):
    """Reflowable stack of display units that keeps message order.

    The source arrangement (vertical, with source offsets) is used unless the
    policy allows the other axis *and* the canvas grew along it by at least
    2x relative to the source aspect: then the units sit side by side,
    bottom-aligned, when that fits the available width. Nested lockups stay
    rigid; only their arrangement changes."""

    def __init__(self, children: list[Block], block_id: str, role: str, label: str, importance: float,
                 src_aspect: float, axes: tuple[str, ...] = ("y", "x"), policy=None):
        self.children = sorted(children, key=lambda b: b.src_box.y)
        self.id, self.role, self.label, self.importance = block_id, role, label, importance
        self.members = [m for c in self.children for m in c.members]
        self.src_aspect, self.axes, self.policy = src_aspect, tuple(axes), policy
        self.kind = getattr(policy, "kind", "display_stack")
        U = self.src_box
        self.offsets = [(c.src_box.x - U.x, c.src_box.y - U.y) for c in self.children]
        gaps = [max(0.0, b.src_box.y - a.src_box.y1) for a, b in zip(self.children, self.children[1:])]
        self.gap = sum(gaps) / len(gaps) if gaps else 0.0

    @property
    def src_box(self) -> Box:
        return union_all(c.src_box for c in self.children)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["kind"] = self.kind
        d["children"] = [c.describe() for c in self.children]
        if self.policy is not None:
            d["policy"] = self.policy.group_id
        return d

    def _vertical(self, s: float, max_w: float, ctx: Ctx):
        laid = [c.layout(s, max_w, ctx) for c in self.children]
        xs, ys = [], []
        y = 0.0
        for i, (l, (ox, oy)) in enumerate(zip(laid, self.offsets)):
            x = min(max(0.0, ox * s), max(0.0, max_w - l.w))
            xs.append(x)
            ys.append(y)
            y += l.h + (self.gap * s if i < len(laid) - 1 else 0.0)
        W = max(x + l.w for x, l in zip(xs, laid))
        return laid, list(zip(xs, ys)), W, y

    def _horizontal(self, s: float, max_w: float, ctx: Ctx):
        laid = [c.layout(s, max_w, ctx) for c in self.children]
        gap = max(2.0, 0.25 * min(l.h for l in laid))
        H = max(l.h for l in laid)
        pos, x = [], 0.0
        for l in laid:
            pos.append((x, H - l.h))
            x += l.w + gap
        return laid, pos, x - gap, H

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        laid, pos, W, H = self._vertical(s, max_w, ctx)
        grew = (ctx.cw / max(1e-6, ctx.ch)) / max(1e-6, self.src_aspect)
        if "x" in self.axes and grew >= 2.0 and len(self.children) > 1:
            hl, hp, hW, hH = self._horizontal(s, max_w, ctx)
            if hW <= max_w + 0.5:
                laid, pos, W, H = hl, hp, hW, hH

        def place(x: float, y: float):
            patches: Patches = {}
            boxes: list[tuple[str, Box]] = []
            for l, (dx, dy) in zip(laid, pos):
                p, b = l.place(x + dx, y + dy)
                patches.update(p)
                boxes.extend(b)
            return patches, boxes + [(self.id, Box(x, y, W, H))]

        return Laid(W, H, place)


class CardBlock(Block):
    """A support backplate (flat panel or shape) with the copy it carries.

    The panel follows its children: children keep their source arrangement
    (vertical, source offsets) or sit side by side when the canvas grew >= 2x
    along x; the backplate is scaled *uniformly* (raster aspect is preserved)
    to the smallest size covering the content plus scaled source padding.
    Photographs are never cards (they are not stretchable backplates)."""

    def __init__(self, panel: dict[str, Any], children: list[Block], block_id: str, role: str, label: str,
                 importance: float, src_aspect: float, policy=None):
        self.panel, self.id, self.role, self.label, self.importance = panel, block_id, role, label, importance
        self.children = sorted(children, key=lambda b: (b.src_box.y, b.src_box.x))
        self.members = [panel] + [m for c in self.children for m in c.members]
        self.src_aspect, self.policy, self.kind = src_aspect, policy, "card"
        P, C = element_box(panel), union_all(c.src_box for c in self.children)
        self.pads = tuple(max(0.0, v) for v in (C.x - P.x, C.y - P.y, P.x1 - C.x1, P.y1 - C.y1))
        self.offsets = [(c.src_box.x - C.x, c.src_box.y - C.y) for c in self.children]
        self.content = C
        self.ow, self.oh = max(1e-6, P.w), max(1e-6, P.h)
        self.flexible = panel["type"] == "shape"

    @property
    def src_box(self) -> Box:
        return element_box(self.panel).union(self.content)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["kind"] = "card"
        d["support"] = self.panel["id"]
        d["children"] = [c.describe() for c in self.children]
        return d

    def _arrange(self, s: float, inner_w: float, ctx: Ctx, horizontal: bool):
        laid = [c.layout(s, inner_w, ctx) for c in self.children]
        pos = []
        if horizontal:
            gap = max(2.0, 0.25 * min(l.h for l in laid))
            H = max(l.h for l in laid)
            x = 0.0
            for l in laid:
                pos.append((x, (H - l.h) / 2))
                x += l.w + gap
            return laid, pos, x - gap, H
        for i, (l, (ox, oy)) in enumerate(zip(laid, self.offsets)):
            if i == 0:
                y = oy * s
            else:
                prev, cur = self.children[i - 1].src_box, self.children[i].src_box
                if cur.y >= prev.y1 - 0.5:  # stacked below: keep the source gap, never overlap
                    # Source gaps inside a card are kept, but never wider than the
                    # adjacent content (an empty band left by an element that is
                    # not part of the card, e.g. an illustration placed elsewhere).
                    gap = min((cur.y - prev.y1) * s, max(l.h, laid[i - 1].h))
                    y = pos[-1][1] + laid[i - 1].h + gap
                else:  # side by side in the source: keep the relative offset
                    y = pos[-1][1] + (oy - self.offsets[i - 1][1]) * s
            pos.append((min(max(0.0, ox * s), max(0.0, inner_w - l.w)), max(0.0, y)))
        W = max(x + l.w for (x, _), l in zip(pos, laid))
        H = max(yy + l.h for (_, yy), l in zip(pos, laid))
        return laid, pos, W, H

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        l0, t0, r0, b0 = self.pads
        inner = max(1.0, max_w - (l0 + r0) * s)
        grew = (ctx.cw / max(1e-6, ctx.ch)) / max(1e-6, self.src_aspect)

        def option(horizontal: bool):
            laid, pos, W, H = self._arrange(s, inner, ctx, horizontal)
            pl, pt, pr, pb = (min(p * s, 0.5 * (W if i % 2 == 0 else H)) for i, p in enumerate((l0, t0, r0, b0)))
            need_w, need_h = W + pl + pr, H + pt + pb
            if self.flexible:
                pw, ph = need_w, need_h
            else:
                k = max(need_w / self.ow, need_h / self.oh)
                pw, ph = self.ow * k, self.oh * k
            fits = pw <= max_w + 0.5 and pw <= ctx.cw and ph <= ctx.ch and W <= inner + 0.5
            return fits, (laid, pos, W, H, pl, need_w, need_h, pw, ph)

        opts = [option(False)]
        if grew >= 2.0 and len(self.children) > 1:
            opts.append(option(True))
        covering = [o for ok, o in opts if ok]
        # Prefer an arrangement the backplate can cover; between covering
        # ones, the shorter one when the canvas grew sideways, else the source axis.
        if covering:
            chosen = min(covering, key=lambda o: o[8]) if grew >= 2.0 else covering[0]
            loose = False
        else:
            # Nothing fits at this scale: report the smallest covering
            # arrangement so the solver shrinks the scale. Copy never spills
            # off its backplate (it would land on an unrelated background).
            chosen = min((o for _, o in opts), key=lambda o: max(o[7] / max(1e-6, min(max_w, ctx.cw)), o[8] / max(1e-6, ctx.ch)))
            loose = False
        laid, pos, W, H, pl, need_w, need_h, pw, ph = chosen
        BW, BH = max(pw, need_w if loose else pw), max(ph, need_h if loose else ph)
        ex, ey = max(0.0, pw - W), max(0.0, ph - H)
        cx = ex * (l0 / (l0 + r0)) if l0 + r0 > 0 else ex / 2
        cy = ey * (t0 / (t0 + b0)) if t0 + b0 > 0 else ey / 2
        if loose:
            cx, cy = min(cx, pl), (BH - H) / 2
        pid = self.panel["id"]

        def place(x: float, y: float):
            patches: Patches = {pid: {"x": x, "y": y + (BH - ph) / 2, "width": pw, "height": ph}}
            boxes: list[tuple[str, Box]] = []
            for lo, (dx, dy) in zip(laid, pos):
                p, b = lo.place(x + cx + dx, y + cy + dy)
                patches.update(p)
                boxes.extend(b)
            return patches, boxes + [(self.id, Box(x, y, BW, BH))]

        return Laid(BW, BH, place)


class GridBlock(Block):
    """Support cards whose panels tile the source canvas, kept as one reflowable
    unit. Arrangement: the source tiling (relative offsets, overlaps resolved
    along the source relation) when the aspect is similar; a single row when
    the canvas is >= 2x wider relative to the source; a single column when it
    is >= 2x taller. Each card keeps its own (covering) layout, so copy stays on
    its backplate and readable copy is not traded for a rigid tiling."""

    def __init__(self, children: list[Block], block_id: str, role: str, label: str, importance: float,
                 src_aspect: float, policy=None):
        self.children = sorted(children, key=lambda b: (b.src_box.x, b.src_box.y))
        self.id, self.role, self.label, self.importance = block_id, role, label, importance
        self.members = [m for c in self.children for m in c.members]
        self.src_aspect, self.policy, self.kind = src_aspect, policy, "card_grid"
        U = self.src_box
        self.offsets = [(c.src_box.x - U.x, c.src_box.y - U.y) for c in self.children]

    @property
    def src_box(self) -> Box:
        return union_all(c.src_box for c in self.children)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["kind"] = "card_grid"
        d["children"] = [c.describe() for c in self.children]
        return d

    def _line(self, laid, horizontal):
        gap = 2.0
        pos, t = [], 0.0
        if horizontal:
            H = max(l.h for l in laid)
            for l in laid:
                pos.append((t, (H - l.h) / 2))
                t += l.w + gap
            return pos, t - gap, H
        W = max(l.w for l in laid)
        order = sorted(range(len(laid)), key=lambda i: (self.children[i].src_box.y, self.children[i].src_box.x))
        pos = [None] * len(laid)
        for i in order:
            pos[i] = ((W - laid[i].w) / 2, t)
            t += laid[i].h + gap
        return pos, W, t - gap

    def _source(self, laid, s):
        placed: list[Box] = []
        pos = []
        for i, (l, (ox, oy)) in enumerate(zip(laid, self.offsets)):
            x, y = ox * s, oy * s
            for j, b in enumerate(placed):
                box = Box(x, y, l.w, l.h)
                if box.intersection_area(b) <= 0:
                    continue
                a0, b0 = self.children[j].src_box, self.children[i].src_box
                if b0.x >= a0.x1 - 1:      # beside in the source: push right
                    x = b.x1
                elif b0.y >= a0.y1 - 1:    # below in the source: push down
                    y = b.y1
            placed.append(Box(x, y, l.w, l.h))
            pos.append((x, y))
        return pos, max(b.x1 for b in placed), max(b.y1 for b in placed)

    def layout(self, s: float, max_w: float, ctx: Ctx) -> Laid:
        ratio = (ctx.cw / max(1e-6, ctx.ch)) / max(1e-6, self.src_aspect)
        if ratio >= 2.0:
            laid = [c.layout(s, max_w, ctx) for c in self.children]
            pos, W, H = self._line(laid, True)
            if W > max_w + 0.5:  # a row that does not fit wraps into the source tiling
                pos, W, H = self._source(laid, s)
        elif ratio <= 0.5:
            laid = [c.layout(s, max_w, ctx) for c in self.children]
            pos, W, H = self._line(laid, False)
        else:
            laid = [c.layout(s, max_w, ctx) for c in self.children]
            pos, W, H = self._source(laid, s)

        def place(x: float, y: float):
            patches: Patches = {}
            boxes: list[tuple[str, Box]] = []
            for l, (dx, dy) in zip(laid, pos):
                p, b = l.place(x + dx, y + dy)
                patches.update(p)
                boxes.extend(b)
            return patches, boxes + [(self.id, Box(x, y, W, H))]

        return Laid(W, H, place)
