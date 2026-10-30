"""Combine the analyzer's rule roles with the learned classifier's roles.

    from engine.role_merge import merged_roles, apply_to_analysis
    roles = merged_roles(source, analysis.roles)          # {id: role}
    apply_to_analysis(analysis, source)                    # rewrites roles/units in place

Nothing here edits engine/analyze.py: callers that want the model's opinion
run `analyze()` as usual and then pass the result through `apply_to_analysis`
(no-op unless `CHAMELEON_ROLE_MODEL=1`, or `force=True`).

Merge policy ("rules are the floor, the model votes on what the rules guess"):

* Structural roles stay with the rules: background, panel and decorative are
  geometric facts (covers the canvas / big flat shape / faint or bleeding)
  that the layouts rely on; the model never changes them, nor promotes
  anything into them.
* A group keeps one role for all its members. The model's vote for a group is
  the probability mass of its text members (a button is named by its label);
  only cta / badge / image-ish group roles are considered.
* The model overrides a text / image / shape role only when it is confident
  (`MIN_CONF`) and clearly ahead of the rule's own role under the model
  (`MIN_MARGIN`), and only inside the element's type family (a text never
  becomes a logo).
* Design-level consistency after the vote: exactly one headline among free
  texts (the one with the highest P(headline); the previous headline becomes
  subhead), at most one hero and one logo among images (highest P), at most
  one CTA (the model's best, else the rules').
* `CHAMELEON_ROLE_MODEL=0` (default) returns the rule roles untouched.
"""

from __future__ import annotations

import os
from typing import Any

from engine import role_model as RM

MIN_CONF = 0.55       # model must be at least this sure of its role
MIN_MARGIN = 0.25     # ... and ahead of the rule role's probability by this much
# The rules' headline / CTA / logo / fine-print decisions fire on specific
# structure and are precise when they fire (legal copy at 12 px is fine print
# even when it reads like body): the model needs to be this sure to take them away.
# (It can still *add* a better headline: the consistency pass then keeps the
# one with the higher P(headline).)
STICKY = {"headline": 0.75, "cta": 0.75, "logo": 0.80, "fine_print": 0.75}
FROZEN = {"background", "panel", "decorative"}     # the rules own these
TEXT_ROLES = {"headline", "subhead", "body", "cta", "badge", "fine_print", "text"}
IMAGE_ROLES = {"logo", "hero", "image"}
SHAPE_ROLES = {"shape"}


def enabled(force: bool | None = None) -> bool:
    if force is not None:
        return force
    return RM.enabled()


def merged_roles(source: dict[str, Any], rule_roles: dict[str, str], probs: dict[str, dict[str, float]] | None = None,
                 force: bool | None = None, min_conf: float = MIN_CONF, min_margin: float = MIN_MARGIN,
                 units: list[list[str]] | None = None) -> dict[str, str]:
    """Rule roles with the model's confident disagreements applied. Returns a
    new dict; `rule_roles` is not modified. `units` lists ids that the analyzer
    treats as one piece of furniture beyond `group_id` (a label in its holder
    shape, a host with attachments): they vote and move together like groups."""
    roles = dict(rule_roles)
    if not enabled(force):
        return roles
    if probs is None:
        probs = RM.predict_proba(source)
    if not probs:
        return roles

    elements = list(source.get("elements") or [])
    by_id = {str(el.get("id")): el for el in elements}
    groups: dict[str, list[str]] = {}
    for el in elements:
        gid = el.get("group_id")
        if gid:
            groups.setdefault(str(gid), []).append(str(el["id"]))
    for k, ids in enumerate(units or []):
        ids = [str(i) for i in ids if str(i) in by_id]
        if len(ids) > 1 and not any(i in g for g in groups.values() for i in ids):
            groups[f"unit:{k}"] = ids
    grouped = {eid for ids in groups.values() for eid in ids}

    # 1. free elements
    for el in elements:
        eid = str(el["id"])
        if eid in grouped or eid not in probs:
            continue
        rule = roles.get(eid)
        if rule in FROZEN or rule is None or el.get("type") == "shape":
            continue        # shapes take their role from the text they hold: the rules' call
        family = _family(str(el.get("type", "")))
        p = {r: v for r, v in probs[eid].items() if r in family}
        if not p:
            continue
        best, conf = max(p.items(), key=lambda t: t[1])
        if best == rule or conf < max(min_conf, STICKY.get(rule, 0.0)) or conf - p.get(rule, 0.0) < min_margin:
            continue
        roles[eid] = best

    # 2. groups: vote with the text members (or all members when there is no text)
    for gid, ids in groups.items():
        rule = roles.get(ids[0])
        if rule in FROZEN or rule is None:
            continue
        voters = [i for i in ids if by_id[i].get("type") == "text"] or ids
        acc: dict[str, float] = {}
        n = 0
        for i in voters:
            if i not in probs:
                continue
            n += 1
            for r, v in probs[i].items():
                acc[r] = acc.get(r, 0.0) + v
        if not n:
            continue
        has_text = any(by_id[i].get("type") == "text" for i in voters)
        family = _family("text") if has_text else _family(str(by_id[voters[0]].get("type", "")))
        p = {r: v / n for r, v in acc.items() if r in family}
        if not p:
            continue
        best, conf = max(p.items(), key=lambda t: t[1])
        if best == rule or conf < max(min_conf, STICKY.get(rule, 0.0)) or conf - p.get(rule, 0.0) < min_margin:
            continue
        mixed = has_text and any(by_id[i].get("type") != "text" for i in ids)
        if mixed and rule in ("badge", "cta") and best not in ("badge", "cta"):
            continue   # a shape + label is a button / chip even if its text reads like copy
        for i in ids:
            roles[i] = best

    _make_consistent(elements, roles, probs, grouped, groups)
    return roles


def _family(etype: str) -> set[str]:
    if etype == "text":
        return TEXT_ROLES - {"text"}
    if etype == "image":
        return IMAGE_ROLES
    return SHAPE_ROLES


def _make_consistent(elements: list[dict[str, Any]], roles: dict[str, str], probs: dict[str, dict[str, float]],
                     grouped: set[str], groups: dict[str, list[str]]) -> None:
    def p(eid: str, role: str) -> float:
        return probs.get(eid, {}).get(role, 0.0)

    free_text = [str(el["id"]) for el in elements
                 if el.get("type") == "text" and str(el["id"]) not in grouped and roles.get(str(el["id"])) in TEXT_ROLES]
    # exactly one headline among free texts
    if free_text:
        heads = [i for i in free_text if roles[i] == "headline"]
        if len(heads) != 1:
            cands = [i for i in free_text if roles[i] not in ("fine_print", "cta")] or free_text
            best = max(cands, key=lambda i: (p(i, "headline"), _font(elements, i)))
            for i in heads:
                if i != best:
                    roles[i] = "subhead"
            roles[best] = "headline"
    # at most one hero and one logo among free images
    for role in ("hero", "logo"):
        have = [str(el["id"]) for el in elements if roles.get(str(el["id"])) == role
                and el.get("type") == "image" and str(el["id"]) not in grouped]
        if len(have) > 1:
            best = max(have, key=lambda i: p(i, role))
            for i in have:
                if i != best:
                    roles[i] = "image"
    # at most one CTA (free text or group)
    cta_units: list[tuple[str, float]] = []
    for i in free_text:
        if roles[i] == "cta":
            cta_units.append((i, p(i, "cta")))
    for gid, ids in groups.items():
        if roles.get(ids[0]) == "cta":
            texts = [i for i in ids if i in probs and any(el.get("type") == "text" for el in elements if str(el["id"]) == i)]
            score = max((p(i, "cta") for i in texts), default=0.0)
            cta_units.append((f"group:{gid}", score))
    if len(cta_units) > 1:
        keep = max(cta_units, key=lambda t: t[1])[0]
        for key, _ in cta_units:
            if key == keep:
                continue
            if key.startswith("group:"):
                for i in groups[key[6:]]:
                    roles[i] = "badge"
            else:
                roles[key] = "body"


def _font(elements: list[dict[str, Any]], eid: str) -> float:
    for el in elements:
        if str(el["id"]) == eid:
            return float(el.get("font_size", 0.0) or 0.0)
    return 0.0


def apply_to_analysis(analysis: Any, source: dict[str, Any], force: bool | None = None) -> dict[str, str]:
    """Rewrite `analysis.roles` and every unit's role / importance with the
    merged roles. Returns {id: (old role) } for the elements that changed.
    Unit membership, anchors and reading order are untouched."""
    if not enabled(force):
        return {}
    units = [list(u.ids) for u in analysis.units if len(u.ids) > 1]
    merged = merged_roles(source, analysis.roles, force=force, units=units)
    changed = {eid: analysis.roles[eid] for eid, r in merged.items() if analysis.roles.get(eid) != r}
    if not changed:
        return {}
    analysis.roles.update(merged)
    try:
        from engine.analyze import IMPORTANCE
    except Exception:  # noqa: BLE001
        IMPORTANCE = {}
    for u in analysis.units:
        new = merged.get(u.ids[0], u.role)
        if new != u.role:
            u.role = new
            u.importance = IMPORTANCE.get(new, u.importance)
            for i in u.ids:
                analysis.roles[i] = new
    heads = [u for u in analysis.units if u.role == "headline"]
    if heads and hasattr(analysis, "headline_font_size"):
        fonts = [float(el.get("font_size", 0.0) or 0.0) for el in source.get("elements", [])
                 if str(el.get("id")) in heads[0].ids and el.get("type") == "text"]
        if fonts:
            analysis.headline_font_size = max(fonts)
    return changed


def summary(source: dict[str, Any], rule_roles: dict[str, str]) -> list[dict[str, Any]]:
    """Side-by-side view for the playground / docs: per element the rule
    role, the model's role + confidence and the merged role."""
    probs = RM.predict_proba(source)
    merged = merged_roles(source, rule_roles, probs=probs, force=True) if probs else dict(rule_roles)
    rows = []
    for el in source.get("elements", []):
        eid = str(el["id"])
        p = probs.get(eid, {})
        best = max(p.items(), key=lambda t: t[1]) if p else ("", 0.0)
        rows.append({"id": eid, "type": el.get("type"), "rule": rule_roles.get(eid), "model": best[0],
                     "confidence": round(best[1], 3), "merged": merged.get(eid)})
    return rows


def flag_value() -> str:
    return os.environ.get("CHAMELEON_ROLE_MODEL", "0")


# ---------------------------------------------------------------------------
# S1: three voters (rules, learned classifier, VLM)
#
#     from engine.role_merge import apply_votes
#     apply_votes(analysis, source, perception)     # perception from engine.perceive, or None
#
# Precedence, per analyzer unit (a group / holder / host / row votes as one):
#   * hard structure stays with the rules: background, panel, decorative, every
#     shape-only unit, and unit membership itself;
#   * the coarse class (picture / headline / cta / logo / badge / secondary /
#     minor, engine.perceive.COARSE) is the majority of the three; with a
#     three-way split the VLM decides when its confidence is >= VLM_TIEBREAK,
#     else the rules. When the VLM abstains (off, no answer, type conflict,
#     low confidence) the classifier alone follows the S4 policy above
#     (MIN_CONF / MIN_MARGIN / STICKY); when the classifier abstains (below
#     MIN_CONF) the VLM alone needs VLM_TIEBREAK against the rules;
#   * fine splits inside a class (subhead / body / fine_print, hero / image)
#     stay with the rules unless both model votes agree against them;
#   * afterwards one headline, one CTA, one logo, one hero.
# Every decision is recorded with the voter that made it (`by`) for the
# playground and the eval.

from engine import perceive as P

VLM_TIEBREAK = 0.6    # VLM confidence needed to decide a tie against the rules
VLM_MIN = 0.4         # below this a VLM vote is ignored altogether
FINE_IN = {P.PICTURE: ("hero", "image"), P.SECONDARY: ("subhead", "body", "fine_print"),
           P.HEADLINE: ("headline",), P.CTA: ("cta",), P.LOGO: ("logo",), P.BADGE: ("badge",)}
IMPORTANCE_BONUS = 0.3   # consistency pass: the VLM's most important box gets this much, halving down the list
RULE_BONUS = 0.5         # ... and the unit the rules gave the role gets this (the rules are precise when they fire)


def vote_roles(source: dict[str, Any], analysis: Any, probs: dict[str, dict[str, float]] | None = None,
               perception: Any = None, model_on: bool | None = None) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Merged roles for every element plus one decision record per unit.
    `analysis.roles` is not modified. Returns (roles, decisions); decisions is
    empty when neither model voted (nothing to merge)."""
    roles = dict(analysis.roles)
    model_on = enabled() if model_on is None else model_on
    if model_on and probs is None:
        probs = RM.predict_proba(source)
    probs = probs if model_on else {}
    probs = probs or {}
    vlm = perception if (perception is not None and getattr(perception, "roles", None)) else None
    if not probs and vlm is None:
        return roles, {}

    by_id = {str(el.get("id")): el for el in source.get("elements") or []}
    decisions: dict[str, dict[str, Any]] = {}
    for u in analysis.units:
        rule = u.role
        rec: dict[str, Any] = {"role": rule, "rule": rule, "model": None, "vlm": None, "by": "rules"}
        decisions[u.key] = rec
        if rule in FROZEN or rule == "background":
            continue
        fam = P.unit_family(u, by_id)
        if fam == "shape":
            continue
        allowed = set(P.ALLOWED[fam])
        mixed = fam == "text" and any(by_id.get(i, {}).get("type") != "text" for i in u.ids) and rule in ("badge", "cta")
        if mixed:
            allowed = {P.BADGE, P.CTA}
        rule_c = P.coarse_of(rule)

        # classifier vote: mean over the members that name the unit
        model_c = model_fine = None
        model_p = 0.0
        p_rule_c = 0.0
        acc = _unit_probs(u, by_id, probs, fam)
        if acc:
            coarse_p: dict[str, float] = {}
            for r, v in acc.items():
                c = P.coarse_of(r)
                if c in allowed:
                    coarse_p[c] = coarse_p.get(c, 0.0) + v
            if coarse_p:
                model_c, model_p = max(coarse_p.items(), key=lambda t: t[1])
                p_rule_c = coarse_p.get(rule_c, 0.0)
                fine_opts = [r for r in FINE_IN.get(model_c, ()) if r in acc]
                model_fine = max(fine_opts, key=lambda r: acc[r]) if fine_opts else None
                rec["model"] = [model_fine or model_c, round(model_p, 3)]
        model_vote = model_c if model_p >= MIN_CONF else None

        # VLM vote
        vlm_c, vlm_conf, vlm_fine = None, 0.0, None
        if vlm is not None:
            got = vlm.get(u.key)
            if got is not None:
                rec["vlm"] = [got[0], round(got[1], 3)]
                if got[0] in allowed and got[1] >= VLM_MIN:
                    vlm_c, vlm_conf = got
                    vlm_fine = vlm.fine.get(u.key)

        # the coarse class
        c, by = rule_c, "rules"
        if model_vote and vlm_c:
            if model_vote == vlm_c and model_vote != rule_c:
                c, by = model_vote, "model+vlm"
            elif model_vote == rule_c:
                by = "rules+model"
            elif vlm_c == rule_c:
                by = "rules+vlm"
            elif vlm_conf >= VLM_TIEBREAK:
                c, by = vlm_c, "vlm"
        elif model_vote:
            if model_vote != rule_c and model_p >= max(MIN_CONF, STICKY.get(rule, 0.0)) \
                    and model_p - p_rule_c >= MIN_MARGIN:
                c, by = model_vote, "model"
            elif model_vote == rule_c:
                by = "rules+model"
        elif vlm_c:
            if vlm_c == rule_c:
                by = "rules+vlm"
            elif vlm_conf >= VLM_TIEBREAK:
                c, by = vlm_c, "vlm"

        # the fine role inside the class
        fine = _fine_role(c, rule, rule_c, model_fine if model_p >= MIN_CONF else None, vlm_fine if vlm_c else None)
        if fine != rule and by in ("rules", "rules+model", "rules+vlm"):
            by = "model+vlm"
        rec["role"], rec["by"] = fine, by
        for i in u.ids:
            roles[i] = fine

    _consistent3(analysis, roles, decisions, probs, vlm, by_id)
    return roles, decisions


def _unit_probs(u: Any, by_id: dict[str, dict[str, Any]], probs: dict[str, dict[str, float]],
                fam: str) -> dict[str, float]:
    voters = [i for i in u.ids if by_id.get(i, {}).get("type") == fam] or list(u.ids)
    acc: dict[str, float] = {}
    n = 0
    for i in voters:
        if i not in probs:
            continue
        n += 1
        for r, v in probs[i].items():
            acc[r] = acc.get(r, 0.0) + v
    return {r: v / n for r, v in acc.items()} if n else {}


def _fine_role(c: str, rule: str, rule_c: str, model_fine: str | None, vlm_fine: str | None) -> str:
    opts = FINE_IN.get(c)
    if not opts:
        return rule
    if len(opts) == 1:
        return opts[0]
    if rule_c == c:
        # the rules' split stands unless both models agree on another one
        if model_fine and vlm_fine and model_fine == vlm_fine and model_fine in opts and model_fine != rule:
            return model_fine
        return rule
    if model_fine in opts:
        return model_fine
    if vlm_fine in opts:
        return vlm_fine
    return "image" if c == P.PICTURE else "body"


def _consistent3(analysis: Any, roles: dict[str, str], decisions: dict[str, dict[str, Any]],
                 probs: dict[str, dict[str, float]], vlm: Any, by_id: dict[str, dict[str, Any]]) -> None:
    """One headline, one CTA, one logo, one hero across the units."""
    units = {u.key: u for u in analysis.units}
    order = list(getattr(vlm, "importance", []) or []) if vlm is not None else []

    def strength(key: str, role: str) -> tuple[float, float]:
        u = units[key]
        p = max((probs.get(i, {}).get(role, 0.0) for i in u.ids), default=0.0)
        if vlm is not None:
            got = vlm.get(key)
            if got is not None and got[0] == P.coarse_of(role):
                p += got[1]
        if key in order:
            p += IMPORTANCE_BONUS / (order.index(key) + 1)
        if u.role == role:
            p += RULE_BONUS
        fonts = [float(by_id.get(i, {}).get("font_size", 0.0) or 0.0) for i in u.ids]
        area = u.box.w * u.box.h
        return (p, max(fonts) if any(fonts) else area)

    def set_role(key: str, role: str) -> None:
        for i in units[key].ids:
            roles[i] = role
        decisions[key]["role"] = role
        decisions[key]["by"] = "consistency"

    def units_with(role: str, fam: str) -> list[str]:
        return [k for k, u in units.items() if roles.get(u.ids[0]) == role and P.unit_family(u, by_id) == fam
                and u.role not in FROZEN and u.role != "background"]

    heads = units_with("headline", "text")
    if len(heads) > 1:
        keep = max(heads, key=lambda k: strength(k, "headline"))
        for k in heads:
            if k != keep:
                set_role(k, "subhead")
    elif not heads:
        prior = [k for k, u in units.items() if u.role == "headline"]
        if prior:
            set_role(prior[0], "headline")
    ctas = units_with("cta", "text")
    if len(ctas) > 1:
        keep = max(ctas, key=lambda k: strength(k, "cta"))
        for k in ctas:
            if k != keep:
                mixed = any(by_id.get(i, {}).get("type") != "text" for i in units[k].ids)
                set_role(k, "badge" if mixed else "body")
    for role in ("logo", "hero"):
        have = units_with(role, "image")
        if len(have) > 1:
            keep = max(have, key=lambda k: strength(k, role))
            for k in have:
                if k != keep:
                    set_role(k, "image")


def apply_votes(analysis: Any, source: dict[str, Any], perception: Any = None,
                probs: dict[str, dict[str, float]] | None = None, model_on: bool | None = None) -> dict[str, str]:
    """Rewrite `analysis.roles` and the units' role / importance with the
    three-voter merge and record the decisions in `analysis.perception`.
    Returns {element id: old role} for what changed. A no-op (and no record)
    when neither the classifier (CHAMELEON_ROLE_MODEL=1) nor a VLM perception
    is available, so the default pipeline is untouched."""
    roles, decisions = vote_roles(source, analysis, probs=probs, perception=perception, model_on=model_on)
    if not decisions:
        if perception is not None and getattr(perception, "status", None) not in (None, "off"):
            analysis.perception = {"archetype": getattr(perception, "archetype", None),
                                   "vlm": perception.status, "decisions": {}, "by": {}, "changed": 0}
        return {}
    changed = {eid: analysis.roles[eid] for eid, r in roles.items() if analysis.roles.get(eid) != r}
    try:
        from engine.analyze import IMPORTANCE
    except Exception:  # noqa: BLE001
        IMPORTANCE = {}
    if changed:
        analysis.roles.update(roles)
        for u in analysis.units:
            new = roles.get(u.ids[0], u.role)
            if new != u.role:
                u.role = new
                u.importance = IMPORTANCE.get(new, u.importance)
                for part in getattr(u, "parts", []) or []:
                    part.role = new
                    part.importance = IMPORTANCE.get(new, part.importance)
                for i in u.ids:
                    analysis.roles[i] = new
        heads = [u for u in analysis.units if u.role == "headline"]
        if heads and hasattr(analysis, "headline_font_size"):
            fonts = [float(el.get("font_size", 0.0) or 0.0) for el in source.get("elements", [])
                     if str(el.get("id")) in heads[0].ids and el.get("type") == "text"]
            if fonts:
                analysis.headline_font_size = max(fonts)
    by: dict[str, int] = {}
    for rec in decisions.values():
        by[rec["by"]] = by.get(rec["by"], 0) + 1
    analysis.perception = {"archetype": getattr(perception, "archetype", None) if perception is not None else None,
                           "vlm": getattr(perception, "status", "off") if perception is not None else "off",
                           "decisions": decisions, "by": by, "changed": len(changed)}
    return changed
