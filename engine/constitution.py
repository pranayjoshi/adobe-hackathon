"""Shared prompt text: the design constitution and the Tier 3 judge rubric.

One copy, used by the VLM proposer, the judge and training exports, so
models are trained and judged against the same principles
(docs/chameleon-design-principles-for-training.md, section 5).
"""

CONSTITUTION = """You adapt an existing design to a new canvas. You may only move and resize the
given blocks and scale their fonts. Never change, drop, or add copy, fonts,
colours, or images. Every block must appear exactly once, fully on canvas.

Priorities, in order:
1. Legible: all copy readable at target size; nothing clipped; text on calm areas.
2. Clear hierarchy: the headline (or hero) leads; supporting copy, then CTA follow.
   Keep the original size order. Use few size levels.
3. Grouped: keep units together (button = shape + label; caption with image).
   Space within a group is smaller than space between groups.
4. Fit the shape: re-arrange for the new aspect ratio (side-by-side for wide,
   stacked for tall) rather than shrinking the original into it.
5. Images intact: no stretching; crop only backgrounds; keep the subject visible.
6. Tidy: align edges, consistent margins and gaps, balanced visual weight,
   keep important content out of platform UI zones.
7. Faithful: same message, focal point and reading order where the shape allows."""

CRITERIA_TEXT = {
    'legibility': 'all copy readable at this size, nothing clipped, text on calm areas',
    'hierarchy': 'one clear entry point; headline, supporting copy, CTA read in order',
    'grouping': 'related items stay together; spacing separates units',
    'composition': 'uses the target shape well; balanced weight; purposeful white space;\n  not merely the original shrunk to fit',
    'imagery': 'product and logo clear, undistorted, uncovered, sensibly cropped',
    'alignment': 'edges line up, consistent gaps, no near-misses',
    'faithfulness': 'clearly the same design and message, adapted rather than altered',
}
CRITERIA = list(CRITERIA_TEXT)

ARRANGEMENTS = ['source_like', 'stack', 'side_by_side', 'banner_row', 'anchored']

# Proposer output; coordinates are normalised to the target canvas.
PLAN_SCHEMA = {
    'type': 'object', 'required': ['groups'], 'additionalProperties': False,
    'properties': {
        'arrangement': {'type': 'string', 'enum': ARRANGEMENTS},
        'rationale': {'type': 'string'},
        'groups': {'type': 'array', 'items': {
            'type': 'object', 'required': ['id', 'x', 'y', 'w', 'h'], 'additionalProperties': False,
            'properties': {'id': {'type': 'string'}, **{k: {'type': 'number', 'minimum': 0, 'maximum': 1} for k in ('x', 'y', 'w', 'h')}}}},
    },
}

# These constrain syntax; the context-aware runtime validator additionally
# checks exact IDs/coverage, group unions, locks and crop eligibility.
PLAN_SCHEMA_V2 = {
    'type': 'object', 'required': ['schema', 'groups', 'elements'], 'additionalProperties': False,
    'properties': {
        'schema': {'type': 'string', 'const': 'layout-plan/2'},
        'arrangement': {'type': 'string', 'enum': ARRANGEMENTS},
        'rationale': {'type': 'string'},
        'groups': PLAN_SCHEMA['properties']['groups'],
        'elements': {'type': 'array', 'items': {
            'type': 'object', 'required': ['id', 'x', 'y', 'w', 'h'], 'additionalProperties': False,
            'properties': {
                'id': {'type': 'string'},
                **{k: {'type': 'number', 'minimum': -1, 'maximum': 2} for k in ('x', 'y')},
                **{k: {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 4} for k in ('w', 'h')},
                'font_size': {'type': 'integer', 'minimum': 1},
                'fit': {'type': 'string', 'const': 'cover'},
            }}},
    },
}

EDIT_ACTION_SCHEMA = {
    'type': 'object', 'required': ['schema', 'operation', 'patch'], 'additionalProperties': False,
    'properties': {
        'schema': {'type': 'string', 'const': 'edit-action/1'},
        'operation': {'type': 'string'},
        'affected_ids': {'type': 'array', 'items': {'type': 'string'}},
        'intended_improvement': {'type': 'string'},
        'patch': {'type': 'object', 'minProperties': 1, 'additionalProperties': {
            'type': 'object', 'minProperties': 1, 'additionalProperties': False,
            'properties': {
                **{k: {'type': 'number'} for k in ('x', 'y')},
                **{k: {'type': 'number', 'exclusiveMinimum': 0} for k in ('width', 'height')},
                'font_size': {'type': 'integer', 'minimum': 1},
                'fit': {'type': 'string', 'const': 'cover'},
            }}},
    },
}

SCORE_SCHEMA = {
    'type': 'object', 'required': ['scores', 'reasons', 'defects', 'overall'], 'additionalProperties': False,
    'properties': {
        'scores': {'type': 'object', 'required': CRITERIA, 'additionalProperties': False, 'properties': {k: {'type': 'integer'} for k in CRITERIA}},
        'reasons': {'type': 'object', 'required': CRITERIA, 'additionalProperties': False, 'properties': {k: {'type': 'string'} for k in CRITERIA}},
        'defects': {'type': 'array', 'items': {'type': 'string'}},
        'overall': {'type': 'integer'},
    },
}

PAIR_SCHEMA = {
    'type': 'object', 'required': ['verdict', 'reason'], 'additionalProperties': False,
    'properties': {'verdict': {'type': 'string', 'enum': ['A', 'B', 'tie', 'unsure']}, 'reason': {'type': 'string'}},
}

_FRAME = """You are a senior graphic designer reviewing an automatic resize of a design.
The ORIGINAL is {src_w}x{src_h}. The resize targets "{format_name}" ({tgt_w}x{tgt_h}).
The resize may only move and scale existing elements and change font sizes. It may
not change copy, fonts, colours or images. All copy must still be present."""

SCORE_STATE = _FRAME + """
Image 1 is the ORIGINAL. Image 2 is the RESIZED version."""

SCORE_PROMPT = SCORE_STATE + """

Score the RESIZED version from 1 to 10 on each criterion: 10 = flawless,
7 = acceptable with minor issues, 4 = obvious shortcomings, 1-2 = broken.
""" + '\n'.join(f'- {k}: {v}' for k, v in CRITERIA_TEXT.items()) + """
For each score name the specific element or area that drove it. List up to 3
concrete defects, most severe first. "overall" is your 1-10 overall judgement."""

PAIR_PROMPT = _FRAME + """
Image 1 is the ORIGINAL. Image 2 is resize A. Image 3 is resize B.
Which resize would a professional designer ship for this format? Weigh, in
order: legibility, hierarchy, grouping, fit to the shape, imagery, polish,
faithfulness. Answer "tie" if they are equivalent and "unsure" if you cannot tell."""


def format_name(width, height):
    from .formats import name_for
    return name_for(width, height)
