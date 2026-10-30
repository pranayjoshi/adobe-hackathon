"""Let content grow when the canvas grows: the strip-to-square fix.

Going from a square-ish design to a strip, everything shrinks, and our engine is good at
that. Going the other way (a 1640x400 cover to a 1000x1500 pin) the canvas has about 1.4x
the area, yet our engine's headline came out at 0.93x its source size (Gaurav's: 1.22x).
Two rules written with shrinking in mind cap the growth:

1. `TextBlock.layout` refuses to upscale a text if that adds lines ("prefer the source
   wrap"). A strip's headline is one long line; in a portrait column it can only grow by
   wrapping, so it was pushed back down to its source size or below. Here the source wrap
   is kept only while the column could hold a source line at the new size; when the
   column is narrower than that, the text may take the extra lines it needs (at most
   MAX_EXTRA_LINES more, so a phrase never falls into one word per line).
2. `Env.s_hi`, the largest scale any template may use, is 1.25x the canvas's smaller axis
   ratio, which for strip-to-square is below 1, so growth stopped at 1.25x. It now also
   allows 1.25x the area ratio (sqrt of target area over source area): the extra room a
   squarer canvas gives.

Both are installed onto the engine at run time by `merged.resize.install`; `engine/` is
not edited.
"""
from __future__ import annotations

import math

MAX_EXTRA_LINES = 2


def text_layout(self, s, max_w, ctx):
    """`engine.blocks.TextBlock.layout`, with the source wrap kept only when it still fits."""
    from engine.blocks import Laid, _shrink_for_width
    from engine.canonical import Box
    from engine.typography import measure, tight_width
    s *= ctx.boost.get(self.role, 1.0)
    fs = max(1, round(self.src_fs * s), min(ctx.floor(self.role), int(self.src_fs)))
    fit = _shrink_for_width(self.el, fs, max_w)

    def allowed(size):
        # A source line, scaled to `size`, needs this much width; a narrower column must wrap.
        needed = float(self.el['width']) * size / self.src_fs
        if max_w + 0.5 >= needed:
            return self.src_lines
        return min(self.src_lines + MAX_EXTRA_LINES, math.ceil(self.src_lines * needed / max(1.0, max_w)))

    while fit.font_size > self.src_fs and fit.lines > allowed(fit.font_size):
        fit = measure(self.el, fit.font_size - 1, max_w)
    width = tight_width(self.el, fit)
    el_id = self.el['id']

    def place(x, y):
        patch = {'x': x, 'y': y, 'width': width, 'height': fit.height, 'font_size': fit.font_size}
        return {el_id: patch}, [(self.id, Box(x, y, width, fit.height))]

    return Laid(width, fit.height, place)


def s_hi(self):
    """`engine.layout.Env.s_hi`, also allowing growth by the canvas's area ratio."""
    area = math.sqrt(self.cw * self.ch / (self.A.sw * self.A.sh))
    return self.scale_bias * 1.25 * max(1.0, min(self.cw / self.A.sw, self.ch / self.A.sh), area)


def install() -> None:
    import engine.blocks
    import engine.layout
    if getattr(engine.blocks, '_merged_growth', False):
        return
    engine.blocks.TextBlock.layout = text_layout
    engine.layout.Env.s_hi = property(s_hi)
    engine.blocks._merged_growth = True
