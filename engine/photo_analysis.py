"""Cached subject / quiet-region maps for photographic assets (guide 08).

Everything here reads the ORIGINAL asset pixels and writes nothing: maps are
diagnostics used to choose geometry, never new image content. A map is cached
by the asset's byte hash plus ``ANALYSIS_VERSION`` and map resolution.

The default estimator is deterministic and CPU-cheap: frequency-tuned colour
saliency (Achanta et al. 2009), spectral-residual saliency (Hou & Zhang 2007),
edge density and a weak skin-tone cue, combined into a subject-importance map.
An optional local segmentation model can be registered with
``register_subject_model``; when it is missing, fails or is uncertain the
heuristic path is used and the typed ``status`` says why.

Masks reach the target canvas through exactly the renderer's contain / cover /
fill mapping (``renderer/converter/assets.py::_fit_image``), so text placement
is judged against the pixels that will actually be drawn.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .canonical import ASSETS_DIR, Box

ANALYSIS_VERSION = 'photo-analysis/2'
MODES = ('heuristic', 'subject_aware')
MAP_SIZE = 96          # longest side of the analysis raster
MODEL_MIN_CONFIDENCE = 0.55


# --------------------------------------------------------------------- switch
def photo_mode(explicit: str | None = None) -> str:
    """`heuristic` (legacy behaviour) unless explicitly or env-selected."""
    value = explicit if explicit is not None else os.environ.get('CHAMELEON_PHOTO_ANALYSIS', 'heuristic')
    return value if value in MODES else 'heuristic'


# ---------------------------------------------------------------- model hook
_MODEL: dict[str, Any] = {'fn': None, 'name': None}


def register_subject_model(fn: Callable[[Any], tuple[Any, float]] | None, name: str = 'local-model') -> None:
    """Install an optional local subject segmenter.

    ``fn(rgba_image) -> (mask HxW in [0,1], confidence in [0,1])``. Pass
    ``None`` to remove it. No model is downloaded or loaded implicitly.
    """
    _MODEL['fn'], _MODEL['name'] = fn, (name if fn else None)
    _CACHE.clear()


def model_status() -> str:
    return f"available:{_MODEL['name']}" if _MODEL['fn'] else 'model_unavailable'


# ---------------------------------------------------------------------- maps
@dataclass(frozen=True)
class PhotoMaps:
    asset_hash: str
    version: str
    size: tuple[int, int]                 # analysis raster (w, h)
    original_size: tuple[int, int]        # asset pixels (w, h)
    saliency: np.ndarray = field(repr=False)     # subject importance [0,1]
    texture: np.ndarray = field(repr=False)      # local busyness [0,1]
    luminance: np.ndarray = field(repr=False)    # WCAG relative luminance [0,1]
    alpha: np.ndarray = field(repr=False)        # opaque coverage [0,1]
    subject: np.ndarray = field(repr=False)      # binary important-subject mask
    subject_boxes: tuple[tuple[float, float, float, float], ...]  # normalized x0,y0,x1,y1
    confidence: float
    provider: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {'asset_hash': self.asset_hash, 'version': self.version, 'size': list(self.size),
                'original_size': list(self.original_size), 'subject_boxes': [list(b) for b in self.subject_boxes],
                'confidence': round(self.confidence, 3), 'provider': self.provider, 'status': self.status}


_CACHE: dict[tuple[str, str, int, str], PhotoMaps] = {}
_HASHES: dict[tuple[str, float, int], str] = {}


def _asset_path(asset_id: str | None):
    path = (ASSETS_DIR / str(asset_id or '')).resolve()
    if not asset_id or not path.is_relative_to(ASSETS_DIR.resolve()) or not path.is_file():
        return None
    return path


def asset_hash(asset_id: str | None) -> str | None:
    path = _asset_path(asset_id)
    if path is None:
        return None
    st = path.stat()
    k = (str(path), st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino)
    if k not in _HASHES:
        if len(_HASHES) >= 128:
            _HASHES.pop(next(iter(_HASHES)))
        _HASHES[k] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _HASHES[k]


def _normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return a
    pad = np.pad(a, r, mode='edge')
    c = np.cumsum(np.cumsum(pad, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    h, w = a.shape
    return (c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]) / (k * k)


def _spectral_residual(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    amp = np.log(np.abs(f) + 1e-9)
    phase = np.angle(f)
    residual = amp - _box_blur(amp, 1)
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    return _normalize(_box_blur(sal, 2))


def _relative_luminance(rgb: np.ndarray) -> np.ndarray:
    v = rgb / 255.0
    v = np.where(v <= .04045, v / 12.92, ((v + .055) / 1.055) ** 2.4)
    return .2126 * v[..., 0] + .7152 * v[..., 1] + .0722 * v[..., 2]


def _components(mask: np.ndarray):
    try:
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        return [(labels == i) for i in range(1, n + 1)]
    except ImportError:  # pragma: no cover - scipy ships with the offline image
        return [mask] if mask.any() else []


def heuristic_maps(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(saliency, texture, confidence) from an RGBA uint8 raster."""
    rgb = rgba[..., :3].astype(np.float64)
    alpha = rgba[..., 3].astype(np.float64) / 255.0
    visible = rgb[alpha > .5]
    if not visible.size or float(np.max(np.std(visible, axis=0))) < 1e-6:
        empty = np.zeros(alpha.shape, dtype=np.float64)
        return empty, empty.copy(), 0.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = .299 * r + .587 * g + .114 * b
    opp = np.stack([lum, r - g, (r + g) / 2 - b], -1)
    weight = alpha[..., None]
    mean = (opp * weight).sum((0, 1)) / max(1e-6, weight.sum())
    blurred = np.stack([_box_blur(opp[..., i], 1) for i in range(3)], -1)
    ft = _normalize(np.sqrt(((blurred - mean) ** 2).sum(-1)))
    sr = _spectral_residual(lum / 255.0)
    gy = np.gradient(lum / 255.0, axis=0) if lum.shape[0] > 1 else np.zeros_like(lum)
    gx = np.gradient(lum / 255.0, axis=1) if lum.shape[1] > 1 else np.zeros_like(lum)
    edges = _normalize(_box_blur(np.hypot(gx, gy), 2))
    # Skin tone in YCbCr: a cheap person cue, weighted modestly.
    cb = 128 - .168736 * r - .331264 * g + .5 * b
    cr = 128 + .5 * r - .418688 * g - .081312 * b
    skin = ((cr > 135) & (cr < 175) & (cb > 80) & (cb < 128) & (lum > 50)).astype(np.float64)
    skin = _box_blur(skin, 1)
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    centre = np.exp(-(((xx / max(1, w - 1)) - .5) ** 2 + ((yy / max(1, h - 1)) - .5) ** 2) / .18)
    sal = .4 * ft + .25 * sr + .2 * edges + .15 * skin
    sal = _normalize(_box_blur(sal, 2) * (.8 + .2 * centre)) * alpha
    local_mean = _box_blur(lum / 255.0, 2)
    local_sq = _box_blur((lum / 255.0) ** 2, 2)
    texture = np.clip(np.sqrt(np.maximum(0, local_sq - local_mean ** 2)) * 4 + .5 * edges, 0, 1) * alpha
    hi = sal[sal > np.percentile(sal, 80)]
    lo = sal[sal <= np.percentile(sal, 80)]
    inside = float(hi.mean()) if hi.size else 0.0
    outside = float(lo.mean()) if lo.size else 0.0
    confidence = float(np.clip((inside - outside) / max(1e-6, inside), 0, 1))
    return sal, texture, confidence


def _subject_mask(sal: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, tuple]:
    vals = sal[alpha > .5]
    if vals.size == 0 or float(vals.max()) <= 1e-9:
        return np.zeros_like(sal, bool), ()
    thr = max(float(vals.mean() + .75 * vals.std()), float(np.percentile(vals, 70)))
    mask = (sal >= thr) & (alpha > .5)
    try:
        from scipy import ndimage
        mask = ndimage.binary_closing(mask, iterations=2) & (alpha > .5)
    except ImportError:  # pragma: no cover
        pass
    h, w = mask.shape
    keep = np.zeros_like(mask)
    boxes = []
    for comp in sorted(_components(mask), key=lambda c: -int(c.sum())):
        if comp.sum() < .01 * h * w:
            continue
        keep |= comp
        ys, xs = np.nonzero(comp)
        boxes.append((xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h))
        if len(boxes) >= 4:
            break
    return keep, tuple(boxes)


def analyze_asset(asset_id: str | None, *, mode: str | None = None, avoid: Iterable[tuple[float, float, float, float]] = (),
                  size: int = MAP_SIZE) -> PhotoMaps | None:
    """Maps for an asset, or None when the asset is missing/unreadable.

    ``avoid`` are normalized asset-space boxes where the SOURCE design placed
    copy: designers keep text off the subject, so these regions are evidence
    of low subject importance (source intent).
    """
    path = _asset_path(asset_id)
    if path is None:
        return None
    digest = asset_hash(asset_id)
    avoid = tuple(tuple(round(float(v), 4) for v in b) for b in avoid)
    model = _MODEL['name'] if photo_mode(mode) == 'subject_aware' else None
    key = (digest, ANALYSIS_VERSION, size, f'{photo_mode(mode)}|{model}|{avoid}')
    if key in _CACHE:
        return _CACHE[key]
    try:
        from PIL import Image
        with Image.open(path) as im:
            if im.width > 16384 or im.height > 16384 or im.width * im.height > 32_000_000:
                return None
            im.load()
            maps = analyze_image(im, digest, mode=mode, avoid=avoid, size=size)
    except (OSError, ValueError):
        return None
    if len(_CACHE) > 64:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = maps
    return maps


def analyze_image(image, digest: str, *, mode: str | None = None, avoid=(), size: int = MAP_SIZE) -> PhotoMaps:
    """Maps for an in-memory PIL image (never modified)."""
    from PIL import Image
    original = image.size
    if isinstance(size, bool) or not isinstance(size, int) or not 4 <= size <= 512:
        raise ValueError('analysis size must be an integer in [4, 512]')
    if image.width > 16384 or image.height > 16384 or image.width * image.height > 32_000_000:
        raise ValueError('analysis image exceeds allocation bounds')
    small = image.convert('RGBA')
    small.thumbnail((size, size), Image.Resampling.BILINEAR)
    rgba = np.asarray(small, dtype=np.uint8).copy()
    sal, texture, confidence = heuristic_maps(rgba)
    provider, status = 'heuristic-saliency', 'ok'
    if photo_mode(mode) == 'subject_aware':
        if _MODEL['fn'] is None:
            status = 'fallback:model_unavailable'
        else:
            try:
                mask, conf = _MODEL['fn'](small)
                mask = np.asarray(mask, dtype=np.float64)
                if mask.shape != sal.shape or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)) or not 0 <= float(conf) <= 1:
                    raise ValueError('model output has wrong shape or range')
                if float(conf) < MODEL_MIN_CONFIDENCE:
                    status = f'fallback:model_uncertain({float(conf):.2f})'
                else:
                    sal = np.maximum(np.clip(mask, 0, 1), .5 * sal)
                    confidence, provider, status = float(conf), str(_MODEL['name']), 'ok'
            except Exception as exc:  # optional component: typed fallback, never a crash
                status = f'fallback:model_error({type(exc).__name__})'
    h, w = sal.shape
    alpha = rgba[..., 3].astype(np.float64) / 255.0
    for x0, y0, x1, y1 in avoid:
        sx0, sy0 = max(0, int(x0 * w)), max(0, int(y0 * h))
        sx1, sy1 = min(w, math.ceil(x1 * w)), min(h, math.ceil(y1 * h))
        if sx1 > sx0 and sy1 > sy0:
            sal[sy0:sy1, sx0:sx1] *= .35
    sal = _normalize(sal) * alpha
    subject, boxes = _subject_mask(sal, alpha)
    return PhotoMaps(digest, ANALYSIS_VERSION, (w, h), original, sal, texture,
                     _relative_luminance(rgba[..., :3].astype(np.float64)), alpha, subject, boxes,
                     confidence, provider, status)


# ------------------------------------------------- renderer-exact fit mapping
def fit_mapping(element: dict[str, Any], original_size: tuple[int, int]):
    """Affine asset->canvas map of the renderer for this element.

    Returns ``(sx, sy, ox, oy, clip)`` with canvas = (u*sx + ox, v*sy + oy) for
    asset pixel (u, v), and ``clip`` the canvas box actually painted. Mirrors
    ``renderer/converter/assets.py::_fit_image`` including its rounding.
    """
    iw, ih = original_size
    w, h = max(1, round(float(element['width']))), max(1, round(float(element['height'])))
    x, y = float(element['x']), float(element['y'])
    # cr-renderer draws the fitted w x h raster stretched into the float box.
    kx, ky = float(element['width']) / w, float(element['height']) / h
    fit = element.get('fit', 'contain')
    if fit == 'cover':
        scale = max(w / iw, h / ih)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        left, top = (nw - w) // 2, (nh - h) // 2
        sx, sy = nw / iw * kx, nh / ih * ky
        return sx, sy, x - left * kx, y - top * ky, Box(x, y, float(element['width']), float(element['height']))
    if fit == 'contain':
        scale = min(w / iw, h / ih)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        ox, oy = (w - nw) // 2, (h - nh) // 2
        return nw / iw * kx, nh / ih * ky, x + ox * kx, y + oy * ky, Box(x + ox * kx, y + oy * ky, nw * kx, nh * ky)
    return w / iw * kx, h / ih * ky, x, y, Box(x, y, float(element['width']), float(element['height']))


@dataclass
class CanvasField:
    """Photo maps resampled onto a canvas grid (cell = canvas / grid)."""
    cw: float
    ch: float
    gw: int
    gh: int
    saliency: np.ndarray
    texture: np.ndarray
    luminance: np.ndarray
    coverage: np.ndarray
    subject: np.ndarray
    status: str
    element_id: str | None = None
    _ints: dict = field(default_factory=dict, repr=False)

    @property
    def cell(self) -> tuple[float, float]:
        return self.cw / self.gw, self.ch / self.gh

    def integral(self, name: str) -> np.ndarray:
        if name not in self._ints:
            a = getattr(self, name).astype(np.float64)
            self._ints[name] = np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))
        return self._ints[name]

    def _cells(self, box: Box):
        cx, cy = self.cell
        x0 = int(np.clip(math.floor(box.x / cx), 0, self.gw)); x1 = int(np.clip(math.ceil(box.x1 / cx), 0, self.gw))
        y0 = int(np.clip(math.floor(box.y / cy), 0, self.gh)); y1 = int(np.clip(math.ceil(box.y1 / cy), 0, self.gh))
        return x0, y0, max(x0 + 1, x1) if x0 < self.gw else x0, max(y0 + 1, y1) if y0 < self.gh else y0

    def mean(self, name: str, box: Box) -> float:
        x0, y0, x1, y1 = self._cells(box)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        s = self.integral(name)
        return float(s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]) / ((x1 - x0) * (y1 - y0))

    def subject_overlap(self, box: Box) -> float:
        """Fraction of ``box`` that covers important subject pixels."""
        return self.mean('subject', box)

    def distance(self) -> np.ndarray:
        """Distance (in canvas px) from each cell to the nearest subject cell."""
        if 'dist' not in self._ints:
            cx, cy = self.cell
            if self.subject.any():
                try:
                    from scipy import ndimage
                    d = ndimage.distance_transform_edt(~self.subject.astype(bool), sampling=(cy, cx))
                except ImportError:  # pragma: no cover
                    d = np.full(self.subject.shape, 1e6)
            else:
                d = np.full(self.subject.shape, 1e6)
            self._ints['dist'] = d
        return self._ints['dist']

    def clearance(self, box: Box) -> float:
        x0, y0, x1, y1 = self._cells(box)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(self.distance()[y0:y1, x0:x1].min())


def canvas_field(element: dict[str, Any], maps: PhotoMaps, cw: float, ch: float, cells: int = 64) -> CanvasField | None:
    """Resample ``maps`` through the renderer's fit of ``element`` on the canvas."""
    if abs(float(element.get('rotation') or 0)) > .5:
        return None
    gw = max(4, min(cells, round(cells * cw / max(cw, ch))))
    gh = max(4, min(cells, round(cells * ch / max(cw, ch))))
    sx, sy, ox, oy, clip = fit_mapping(element, maps.original_size)
    iw, ih = maps.original_size
    mw, mh = maps.size
    xs = (np.arange(gw) + .5) * cw / gw
    ys = (np.arange(gh) + .5) * ch / gh
    u = (xs - ox) / sx          # asset px
    v = (ys - oy) / sy
    inside_x = (xs >= clip.x) & (xs < clip.x1) & (u >= 0) & (u < iw)
    inside_y = (ys >= clip.y) & (ys < clip.y1) & (v >= 0) & (v < ih)
    mu = np.clip((u / iw * mw).astype(int), 0, mw - 1)
    mv = np.clip((v / ih * mh).astype(int), 0, mh - 1)
    inside = inside_y[:, None] & inside_x[None, :]
    pick = lambda a: np.where(inside, a[np.ix_(mv, mu)], 0.0)
    coverage = pick(maps.alpha) * float(element.get('opacity', 1.0))
    return CanvasField(cw, ch, gw, gh, pick(maps.saliency), pick(maps.texture), pick(maps.luminance),
                       coverage, (pick(maps.subject.astype(np.float64)) > .5) & (coverage > .5), maps.status, element.get('id'))


# ---------------------------------------------------------- contrast helpers
def color_luminance(value: str | None) -> float:
    from .quality import color, luminance
    return luminance(color(value or '#000000'))


def contrast(l1: float, l2: float) -> float:
    return (max(l1, l2) + .05) / (min(l1, l2) + .05)


@dataclass(frozen=True)
class QuietRegion:
    box: Box
    cost: float
    subject_overlap: float
    clearance: float
    contrast: float
    texture: float

    def to_dict(self):
        return {'box': self.box.to_dict(), 'cost': round(self.cost, 4), 'subject_overlap': round(self.subject_overlap, 4),
                'clearance': round(self.clearance, 2), 'contrast': round(self.contrast, 2), 'texture': round(self.texture, 3)}


def region_cost(f: CanvasField, box: Box, text_lum: float | None = None, prefer: tuple[float, float] | None = None,
                clearance: float = 0.0) -> QuietRegion:
    sal = f.mean('saliency', box); tex = f.mean('texture', box); sub = f.subject_overlap(box)
    lum = f.mean('luminance', box)
    ratio = contrast(text_lum, lum) if text_lum is not None else 21.0
    clear = f.clearance(box)
    cost = 3.0 * sub + 1.2 * sal + .6 * tex
    if text_lum is not None:
        cost += .25 * max(0.0, 4.5 - ratio)
    if clearance > 0:
        cost += .8 * max(0.0, 1 - clear / clearance)
    if prefer is not None:
        cost += .6 * math.hypot((box.cx - prefer[0]) / f.cw, (box.cy - prefer[1]) / f.ch)
    return QuietRegion(box, cost, sub, clear, ratio, tex)


def quiet_regions(f: CanvasField, w: float, h: float, *, text_lum: float | None = None,
                  within: Box | None = None, avoid: Iterable[Box] = (), prefer: tuple[float, float] | None = None,
                  max_subject: float = .06, min_coverage: float = .97, max_results: int = 4) -> list[QuietRegion]:
    """Feasible w x h rectangles ranked by cost (low = quiet, clear, legible).

    Feasible means fully on the photo, inside ``within``, off every ``avoid``
    box and overlapping at most ``max_subject`` of important subject. All
    window sums use integral images (one vectorised pass per map) and the
    clearance term a min-filtered distance transform. An empty list is an
    explicit "no quiet region" result for the caller's fallback.
    """
    within = within or Box(0, 0, f.cw, f.ch)
    if w > within.w + .5 or h > within.h + .5 or w <= 0 or h <= 0:
        return []
    cx, cy = f.cell
    wc, hc = max(1, math.ceil(w / cx - 1e-6)), max(1, math.ceil(h / cy - 1e-6))
    if wc > f.gw or hc > f.gh:
        return []
    def window(name):
        s_ = f.integral(name)
        return (s_[hc:, wc:] - s_[:-hc, wc:] - s_[hc:, :-wc] + s_[:-hc, :-wc]) / (wc * hc)
    sal, tex, sub, lum, cov = (window(n) for n in ('saliency', 'texture', 'subject', 'luminance', 'coverage'))
    ny, nx = sal.shape
    xs = np.arange(nx) * cx; ys = np.arange(ny) * cy
    # Clamp the true box inside within; window covers the cells it touches.
    X = np.clip(np.broadcast_to(xs[None, :], (ny, nx)), within.x, max(within.x, within.x1 - w))
    Y = np.clip(np.broadcast_to(ys[:, None], (ny, nx)), within.y, max(within.y, within.y1 - h))
    ok = (cov >= min_coverage) & (sub <= max_subject)
    ok &= (xs[None, :] + wc * cx >= within.x) & (xs[None, :] <= within.x1)
    ok &= (ys[:, None] + hc * cy >= within.y) & (ys[:, None] <= within.y1)
    for a in avoid:
        ok &= ~((X < a.x1) & (X + w > a.x) & (Y < a.y1) & (Y + h > a.y))
    if not ok.any():
        return []
    clearance = .03 * min(f.cw, f.ch)
    try:
        from scipy import ndimage
        dmin = ndimage.minimum_filter(f.distance(), size=(hc, wc), mode='nearest')[hc // 2: hc // 2 + ny, wc // 2: wc // 2 + nx]
        dmin = np.pad(dmin, ((0, ny - dmin.shape[0]), (0, nx - dmin.shape[1])), mode='edge')
    except ImportError:  # pragma: no cover
        dmin = np.full((ny, nx), 1e6)
    cost = 3.0 * sub + 1.2 * sal + .6 * tex
    ratio = np.full_like(cost, 21.0)
    if text_lum is not None:
        ratio = (np.maximum(text_lum, lum) + .05) / (np.minimum(text_lum, lum) + .05)
        cost = cost + .25 * np.maximum(0.0, 4.5 - ratio)
    cost = cost + .8 * np.maximum(0.0, 1 - dmin / clearance)
    if prefer is not None:
        cost = cost + .6 * np.hypot((X + w / 2 - prefer[0]) / f.cw, (Y + h / 2 - prefer[1]) / f.ch)
    cost = np.where(ok, cost, np.inf)
    order = np.argsort(cost, axis=None, kind='stable')
    picked: list[QuietRegion] = []
    for flat in order[:4000]:
        iy, ix = divmod(int(flat), nx)
        if not np.isfinite(cost[iy, ix]):
            break
        b = Box(float(X[iy, ix]), float(Y[iy, ix]), w, h)
        if any(b.intersection_area(p.box) >= .5 * b.area for p in picked):
            continue
        # Grid windows were ranked before clamping: recheck the actual returned box.
        q = region_cost(f, b, text_lum, prefer, clearance)
        if f.mean('coverage', b) < min_coverage or q.subject_overlap > max_subject:
            continue
        picked.append(q)
        if len(picked) >= max_results:
            break
    return picked


# ------------------------------------------------------ design-level helpers
def photo_hosts(design: dict[str, Any], analysis=None) -> list[dict[str, Any]]:
    """Actual photographic rasters, excluding flat/graphic/transparent overlays.

    Classification uses original asset pixels. A bounding rectangle alone is
    never evidence that a transparent decorative raster is a photograph.
    ``analysis`` is accepted for a common provider/operator interface.
    """
    try:
        from .relations import asset_stats
    except ImportError:  # independent lane: conservative pixel fallback
        asset_stats = None
    cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    out = []
    for e in design['elements']:
        if e.get('type') != 'image' or abs(float(e.get('rotation') or 0)) > .5 or float(e.get('opacity', 1)) < .9:
            continue
        if Box(float(e['x']), float(e['y']), float(e['width']), float(e['height'])).clip(cw, ch).area < .25 * cw * ch:
            continue
        if _asset_path(e.get('asset_id')) is None:
            continue
        if asset_stats is not None:
            if asset_stats(e.get('asset_id')).kind != 'photo':
                continue
        else:
            from .structure import flat_asset
            from .scene import low_detail_asset
            if flat_asset(e.get('asset_id')) or low_detail_asset(e.get('asset_id')):
                continue
        out.append(e)
    return out


def painted_coverage(element: dict[str, Any], box: Box, design: dict[str, Any]) -> float:
    """Approximate alpha coverage through the renderer's contain/cover/fill map.

    Missing pixels and rotated rasters are unverified, and cannot establish a
    support relation. This is a conservative sampled-region estimate.
    """
    if element.get('type') != 'image' or abs(float(element.get('rotation') or 0)) > .5:
        return 0.0
    maps = analyze_asset(element.get('asset_id'), mode='heuristic')
    if maps is None or box.area <= 0:
        return 0.0
    sx, sy, ox, oy, clip = fit_mapping(element, maps.original_size)
    iw, ih = maps.original_size
    mw, mh = maps.size
    xs = box.x + (np.arange(16) + .5) * box.w / 16
    ys = box.y + (np.arange(8) + .5) * box.h / 8
    u, v = (xs - ox) / sx, (ys - oy) / sy
    inside = ((ys[:, None] >= clip.y) & (ys[:, None] < clip.y1) & (xs[None, :] >= clip.x) &
              (xs[None, :] < clip.x1) & (v[:, None] >= 0) & (v[:, None] < ih) & (u[None, :] >= 0) & (u[None, :] < iw))
    mu = np.clip((u / iw * mw).astype(int), 0, mw - 1)
    mv = np.clip((v / ih * mh).astype(int), 0, mh - 1)
    alpha = maps.alpha[np.ix_(mv, mu)] * float(element.get('opacity', 1))
    return float(((alpha >= .9) & inside).mean())


def source_photo_supports(source: dict[str, Any], analysis=None) -> dict[str, str]:
    """Text id -> original photographic host, using C1 graph plus real alpha.

    An available semantic graph is authoritative about support membership.
    Without it, conservative source pixel/geometry inference is used. Both
    paths recheck pixels and opacity, so stale box evidence cannot attach
    copy to a transparent overlay.
    """
    from .canonical import rotated_extent
    photos = {e['id']: e for e in photo_hosts(source, analysis)}
    graph = getattr(analysis, 'graph', None)
    edges = getattr(graph, 'edges', None)
    if edges is None and isinstance(graph, dict):
        edges = graph.get('edges')
    allowed = None if edges is None else {(x.get('a'), x.get('b')) for x in edges if x.get('kind') == 'supports_on_photo'}
    out = {}
    for t in source['elements']:
        if t.get('type') != 'text' or not str(t.get('content', '')).strip():
            continue
        box = rotated_extent(t)
        candidates = [p for p in photos.values() if p.get('z_index', 0) < t.get('z_index', 0)
                      and (allowed is None or (p['id'], t['id']) in allowed)
                      and painted_coverage(p, box, source) >= .9]
        if candidates:
            out[t['id']] = max(candidates, key=lambda p: p.get('z_index', 0))['id']
    return out


def source_intent_boxes(source: dict[str, Any], host: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    """Source copy regions over ``host`` in normalized asset coordinates."""
    from .canonical import rotated_extent
    info = analyze_asset(host.get('asset_id'), mode='heuristic')
    if info is None:
        return []
    sx, sy, ox, oy, _ = fit_mapping(host, info.original_size)
    iw, ih = info.original_size
    out = []
    for t in source['elements']:
        if t.get('type') != 'text' or t.get('z_index', 0) <= host.get('z_index', 0) or not str(t.get('content', '')).strip():
            continue
        b = rotated_extent(t)
        u0, v0 = (b.x - ox) / sx / iw, (b.y - oy) / sy / ih
        u1, v1 = (b.x1 - ox) / sx / iw, (b.y1 - oy) / sy / ih
        u0, v0, u1, v1 = max(0, u0), max(0, v0), min(1, u1), min(1, v1)
        if u1 > u0 and v1 > v0:
            out.append((u0, v0, u1, v1))
    return out


def design_field(source: dict[str, Any], design: dict[str, Any], host_id: str, *, mode: str | None = None,
                 cells: int = 64) -> CanvasField | None:
    """Canvas field for the photo ``host_id`` as placed in ``design``."""
    host = next((e for e in design['elements'] if e['id'] == host_id), None)
    original = next((e for e in source['elements'] if e['id'] == host_id), None)
    if host is None or original is None:
        return None
    maps = analyze_asset(host.get('asset_id'), mode=mode, avoid=source_intent_boxes(source, original))
    if maps is None:
        return None
    return canvas_field(host, maps, float(design['canvas']['width']), float(design['canvas']['height']), cells)


def text_on_subject(source: dict[str, Any], design: dict[str, Any], *, mode: str | None = None) -> dict[str, float]:
    """Per text id: fraction of its ink box lying on an important photo subject."""
    from .quality import ink
    out: dict[str, float] = {}
    for host in photo_hosts(design):
        f = design_field(source, design, host['id'], mode=mode)
        if f is None:
            continue
        for t in design['elements']:
            if t.get('type') != 'text' or t.get('z_index', 0) <= host.get('z_index', 0) or not str(t.get('content', '')).strip():
                continue
            v = f.subject_overlap(ink(t))
            if v > 0:
                out[t['id']] = max(out.get(t['id'], 0.0), round(v, 4))
    return out
