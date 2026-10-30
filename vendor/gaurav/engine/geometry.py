"""Tiny axis-aligned box helpers shared by the analyzer and the layout code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def of(cls, el: dict[str, Any]) -> "Box":
        return cls(float(el["x"]), float(el["y"]), float(el["width"]), float(el["height"]))

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def intersection_area(self, other: "Box") -> float:
        iw = min(self.right, other.right) - max(self.x, other.x)
        ih = min(self.bottom, other.bottom) - max(self.y, other.y)
        if iw <= 0 or ih <= 0:
            return 0.0
        return iw * ih

    def overlaps(self, other: "Box", tol: float = 0.0) -> bool:
        return (
            self.x + tol < other.right
            and other.x + tol < self.right
            and self.y + tol < other.bottom
            and other.y + tol < self.bottom
        )

    def contains_fraction(self, other: "Box") -> float:
        """Fraction of `other`'s area that lies inside this box."""
        if other.area <= 0:
            return 0.0
        return self.intersection_area(other) / other.area

    @staticmethod
    def union(boxes: Iterable["Box"]) -> "Box":
        boxes = list(boxes)
        x0 = min(b.x for b in boxes)
        y0 = min(b.y for b in boxes)
        x1 = max(b.right for b in boxes)
        y1 = max(b.bottom for b in boxes)
        return Box(x0, y0, x1 - x0, y1 - y0)

    def scaled(self, s: float) -> "Box":
        return Box(self.x * s, self.y * s, self.w * s, self.h * s)

    def copy(self) -> "Box":
        return Box(self.x, self.y, self.w, self.h)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
