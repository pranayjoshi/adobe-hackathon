"""Finite, renderer-compatible text choices for joint layout search.

No text or style runs are rewritten. The cache includes actual font bytes and
the measurement profile; estimates and substituted fonts are explicitly marked.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from typing import Any, Iterable
from types import MappingProxyType

from .typography import measure, measurement_identity, _metrics, _style_key


@dataclass(frozen=True)
class TextConfiguration:
    box_width: int
    box_height: int
    font_size: int
    lines: int
    widest_advance: float
    baseline_offsets: tuple[float, ...]
    overflow: bool
    renderer_mode: str
    fingerprint: str
    typography_profile: str
    verified: bool
    # Glyph ink bounds need a rendered mask; line metrics cannot certify them.
    ink_bounds: tuple[float, float, float, float] | None = None

    @property
    def width(self) -> int:
        return self.box_width

    @property
    def height(self) -> int:
        return self.box_height


@dataclass(frozen=True)
class GroupLayoutConfiguration:
    """Validated finite group choice; geometry-only patches remain immutable.

    ``anchor`` is an exact target-space origin fixed by child locks. A solver
    may choose an anchored configuration but cannot translate it afterwards.
    """
    id: str
    group_id: str
    width: float
    height: float
    child_patches: Any
    anchor: tuple[float,float] | None
    readability_penalty: int
    policy_version: str

    def place(self,x:float,y:float) -> dict[str,dict[str,float]]:
        if self.anchor is not None:x,y=self.anchor
        return {eid:{**patch,'x':patch['x']+x,'y':patch['y']+y}
                for eid,patch in self.child_patches.items()}


def enumerate_group_configurations(policy, source, analysis, target, *, locks=None,
                                   max_configs=8, allow_estimated=False):
    """Bridge semantic policies to solver-ready measured group configurations.

    The historical C1 enumerator is a proposal provider. This boundary checks
    exact membership, actual rotated extents, text fit/resources, image aspect,
    nested rigid transforms, policy-owned support containment and child locks.
    Unsafe proposals are discarded before a solver can certify them.
    """
    from .group_policies import POLICY_VERSION,enumerate_configs
    from .canonical import rotated_extent,union_all
    from .blocks import FLOORS
    by={e['id']:e for e in source['elements']}
    policies={p.group_id:p for p in getattr(analysis,'policies',())}
    policies[policy.group_id]=policy
    pins=locks or {}
    proposals=enumerate_configs(policy,source,analysis,target,max_configs=max_configs,
                                locks=pins,policies=policies.values())
    accepted=[]
    for candidate in proposals:
        if not candidate.feasible or set(candidate.child_patches)!=set(policy.member_ids):continue
        absolute=candidate.place(0,0)
        reconstructed={eid:{**by[eid],**patch} for eid,patch in absolute.items()}
        valid=True;penalty=0
        for eid,element in reconstructed.items():
            patch=absolute[eid]
            if set(patch)-{'x','y','width','height','font_size'} or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                for v in patch.values()):valid=False;break
            if element['width']<=0 or element['height']<=0:valid=False;break
            if eid in pins and any(abs(element.get(k,-1e9)-v)>1e-6 for k,v in pins[eid].items()):valid=False;break
            if element['type']=='image':
                before=by[eid]['width']/by[eid]['height'];after=element['width']/element['height']
                if abs(math.log(after/before))>.00001:valid=False;break
            if element['type']=='text':
                identity=measurement_identity(element)
                if not allow_estimated and not identity['verified']:valid=False;break
                fit=measure(element,int(element['font_size']),element['width'])
                role=getattr(analysis,'roles',{}).get(eid,'body')
                floor=min(int(by[eid]['font_size']),FLOORS.get(role,8))
                if fit.overflow or fit.height>element['height']+.5 or fit.font_size<floor:valid=False;break
                penalty+=max(0,int(by[eid]['font_size'])-fit.font_size)*100+fit.lines-1
        if not valid:continue
        # Every nested rigid subtree must share one scale and translation.
        rigid=[p for p in policies.values() if p.mode=='rigid' and set(p.member_ids)<=set(policy.member_ids)]
        if policy.mode=='rigid' and policy.group_id not in {p.group_id for p in rigid}:rigid.append(policy)
        for subtree in rigid:
            ids=subtree.member_ids;first=ids[0];scale=reconstructed[first]['width']/by[first]['width']
            tx=reconstructed[first]['x']-by[first]['x']*scale;ty=reconstructed[first]['y']-by[first]['y']*scale
            for eid in ids:
                original=by[eid];element=reconstructed[eid]
                if any(abs(element[k]-(original[k]*scale+(tx if k=='x' else ty if k=='y' else 0)))>1e-5
                       for k in ('x','y','width','height')):valid=False;break
            if not valid:break
        if not valid:continue
        for support in (p for p in policies.values() if p.mode=='support' and p.support_id and set(p.member_ids)<=set(policy.member_ids)):
            panel=reconstructed[support.support_id]
            if panel['type']=='image':
                from .relations import asset_stats
                if asset_stats(panel.get('asset_id')).kind=='photo':valid=False;break
            # Transparent panels require actual opaque coverage; boxes alone
            # would admit white copy sitting in a transparent triangle corner.
            from .relations import alpha_coverage,glyph_box
            for eid in support.member_ids:
                if eid==support.support_id or reconstructed[eid]['type']!='text':continue
                if alpha_coverage(panel,glyph_box(reconstructed[eid]))<.95:valid=False;break
            if not valid:break
        if not valid:continue
        bounds=union_all(rotated_extent(element) for element in reconstructed.values())
        if bounds.w>target['width']+1e-6 or bounds.h>target['height']+1e-6:continue
        anchor=(bounds.x,bounds.y) if candidate.anchored else None
        if anchor and (bounds.x < -1e-6 or bounds.y < -1e-6 or bounds.x1>target['width']+1e-6 or bounds.y1>target['height']+1e-6):continue
        patches={eid:MappingProxyType({**patch,'x':patch['x']-bounds.x,'y':patch['y']-bounds.y})
                 for eid,patch in absolute.items()}
        accepted.append(GroupLayoutConfiguration(candidate.id,policy.group_id,bounds.w,bounds.h,
            MappingProxyType(patches),anchor,penalty,POLICY_VERSION))
    return tuple(accepted)


@lru_cache(maxsize=2048)
def _enumerate(serialized: str, fingerprint: str, profile: str, mode: str,
               verified: bool, widths: tuple[int, ...], sizes: tuple[int, ...],
               max_height: float, max_configs: int) -> tuple[TextConfiguration, ...]:
    el = json.loads(serialized)
    choices = []
    for width in widths:
        for size in sizes:
            fit = measure(el, size, width)
            if fit.overflow or math.ceil(fit.height) > max_height:
                continue
            ascent, _, _ = _metrics(el.get("font_family", ""), *_style_key(el), size)
            step = size * float(el.get("line_height") or 1.2)
            choices.append(TextConfiguration(width, math.ceil(fit.height), size, fit.lines,
                fit.line_width, tuple(-ascent + i * step for i in range(fit.lines)), False,
                mode, fingerprint, profile, verified))
    # A configuration is dominated only when another is at least as readable
    # and no larger in either dimension, with no additional wrapped lines.
    frontier = [c for c in choices if not any(
        d.font_size >= c.font_size and d.width <= c.width and d.height <= c.height
        and d.lines <= c.lines and (d.font_size > c.font_size or d.width < c.width
                                   or d.height < c.height or d.lines < c.lines)
        for d in choices)]
    frontier.sort(key=lambda c: (-c.font_size, c.lines, c.width, c.height))
    if len(frontier) > max_configs:
        # Sample the complete tradeoff frontier, preserving both the largest
        # type and the smallest alternatives rather than taking only big type.
        indices = [round(i * (len(frontier)-1)/(max_configs-1)) for i in range(max_configs)] if max_configs > 1 else [0]
        frontier = [frontier[i] for i in indices]
    return tuple(frontier)


def enumerate_text_configurations(
    el: dict[str, Any], widths: Iterable[float] | None = None, *,
    max_width: float | None = None, max_height: float | None = None,
    fs_min: int = 1, fs_max: int | None = None, max_configs: int = 12,
    allow_estimated: bool = True,
) -> tuple[TextConfiguration, ...]:
    """Enumerate a bounded measured Pareto frontier without mutating ``el``.

    Explicit ``widths`` restrict allowed widths. Otherwise try source-related
    columns and word-boundary advances. Integer box dimensions are measured
    *after* quantization so exported geometry reproduces the chosen wrap.
    Missing faces remain inspectable but can be excluded with allow_estimated.
    """
    if el.get("type") != "text":
        raise ValueError("Text configurations require a text element")
    if max_configs < 1:
        return ()
    upper_width = float(max_width if max_width is not None else el["width"])
    upper_height = float(max_height if max_height is not None else math.inf)
    if not math.isfinite(upper_width) or upper_width < 1 or upper_height <= 0:
        return ()
    identity = measurement_identity(el)
    if not allow_estimated and not identity["verified"]:
        return ()
    hi = max(1, int(fs_max if fs_max is not None else el.get("font_size", 12)))
    lo = max(1, int(fs_min))
    if lo > hi:
        return ()
    # A bounded sample plus exact endpoints avoids a font-size-proportional
    # solve construction cost. Include the current size for source fidelity.
    sizes = {lo, hi, min(hi, max(lo, int(el.get("font_size", hi))))}
    sizes.update(round(lo + (hi-lo)*i/7) for i in range(8))
    if widths is None:
        base = float(el["width"])
        offered = [base*f for f in (.5, .75, 1, 1.25, 1.5)] + [upper_width]
        words = str(el.get("content", "")).split(" ")[:24]
        # Prefix advances straddle greedy wrap transitions. Full measurement
        # still makes the authoritative decision, including styled spans.
        for n in range(1, len(words)+1):
            probe = {**el, "content": " ".join(words[:n]), "style_runs": []}
            advance = measure(probe, hi, 1e9).line_width
            offered.extend((advance + 1, advance + hi*.35 + 1))
    else:
        offered = list(widths)
    quantized = tuple(sorted({max(1, math.floor(float(w))) for w in offered
                              if math.isfinite(float(w)) and 1 <= float(w) <= upper_width}))
    typography = {k: el.get(k) for k in ("type", "content", "font_family", "font_weight",
        "font_style", "italic", "capitalize", "letter_spacing", "line_height", "style_runs")}
    return _enumerate(json.dumps(typography, sort_keys=True), identity["fingerprint"],
        identity["profile"], identity["renderer_mode"], identity["verified"], quantized,
        tuple(sorted(sizes, reverse=True)), upper_height, max_configs)
