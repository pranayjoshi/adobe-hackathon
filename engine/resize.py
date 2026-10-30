"""
Competition entry point: `resize(source, target_canvas) -> design`.

The approach (see docs/chameleon-end-to-end-build-plan.md):
  1. analyse the source - roles, explicit/inferred groups, intended overlaps
  2. propose several structurally different arrangements for the target
  3. solve each with renderer-accurate text measurement and scale search
  4. serialise by patching allowlisted geometry onto untouched source elements
  5. validate preservation rules, score quality, return the best valid one
"""

from typing import Any


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    from final.resize import resize as final_resize  # the submission engine; see final/README.md
    return final_resize(source, target_canvas)
