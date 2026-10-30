"""v11 switches: one master flag and one per piece.

    CHAMELEON_V11=0|1            every v11 piece off / on (default: DEFAULT_ON per piece)
    CHAMELEON_V11_<PIECE>=0|1    one piece, overrides the master

Pieces: COMPOSE (banner / wide compositions), SOLVE (joint minimal-move
constraint pass), TYPESCALE (type scale, line measure, whitespace budget),
UNITS (compound phrases, text lockups, secondary images), STRIP (tiny-format
composer), AIR (the whitespace-budget plan variants: measured worse on the
designer references, so opt-in only, the master flag does not turn it on),
VISIBLE (v12, engine/visibility.py: content the source showed is never
hidden under a panel; a repair in realize plus a judge hard fail).
Read per call so tests and the eval tools can flip them; CHAMELEON_V11=0
turns every piece off and the engine is then byte-identical to v10.
"""

from __future__ import annotations

import os

PIECES = ("compose", "solve", "typescale", "units", "strip", "air", "visible")
# v11 defaults (docs/EVAL_V11.md): the five pieces on, the whitespace variants off
DEFAULT_ON = {"compose": True, "solve": True, "typescale": True, "units": True, "strip": True,
              "air": False, "visible": True}

OPT_IN = ("air",)           # pieces only their own flag turns on
_OFF = ("0", "false", "no", "off")


def enabled(piece: str) -> bool:
    own = os.environ.get(f"CHAMELEON_V11_{piece.upper()}", "").strip().lower()
    if own:
        return own not in _OFF
    master = os.environ.get("CHAMELEON_V11", "").strip().lower()
    if master and piece not in OPT_IN:
        return master not in _OFF
    return DEFAULT_ON.get(piece, False)
