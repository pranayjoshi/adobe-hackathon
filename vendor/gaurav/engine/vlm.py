"""Local vision-language model adapter (llama.cpp via llama-cpp-python).

The engine may ask a small VLM (Qwen3-VL GGUF, baked into the Docker image
at build time, see the Dockerfile) to look at a rendered design and answer a
question as JSON. This module is the only place that knows how to do that;
nothing in the engine depends on it succeeding:

  * `describe(image, prompt, schema)` (one PIL image or a list of them)
    returns a dict that validates against `schema`, or None. It never raises into the caller: a missing model, a
    broken backend, a timeout or unparsable output all return None.
  * `CHAMELEON_VLM=0` switches it off (describe() returns None at once, no
    model is loaded, no native code runs). Unset or `1` means "on when a
    model file is found" (`CHAMELEON_VLM_MODEL` / `CHAMELEON_VLM_MMPROJ`,
    else the first text + mmproj GGUF pair under `models/`).
  * The model is loaded lazily on the first call and kept for the process.
  * Decoding is deterministic: temperature 0, fixed seed, no repeat penalty,
    and the output is constrained to the JSON schema with a llama.cpp
    grammar, so the model cannot produce anything but a JSON object.
  * Images are downscaled to <= MAX_SIDE px on the long side (Qwen3-VL uses
    32 px patches: 672 px -> ~441 image tokens) and PNG-encoded; the result
    is cached by (image hash, prompt hash) in memory and, when
    `CHAMELEON_VLM_CACHE_DIR` is set, on disk.
  * Timeouts are cooperative (no threads, no subprocesses): generation is
    streamed and stopped at the deadline; the uninterruptible part (image
    encode + prompt evaluation, a few seconds) is measured, and a call that
    overshoots the deadline there counts as a timeout. MAX_TIMEOUTS of them
    switch the adapter off for the rest of the process.

Not wired into resize() yet: callers opt in explicitly.

Backends are pluggable (`set_backend`) so tests run with a stub and no
weights: a backend only has to implement `complete(png_bytes, prompt,
schema, max_tokens, deadline) -> str`.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

MAX_SIDE = 672                 # long side of the image sent to the model
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_SEED = 7
DEFAULT_N_CTX = 4096
MAX_TIMEOUTS = 2               # per process, then the adapter switches itself off
MAX_FAILURES = 2               # backend exceptions, same
CACHE_ENTRIES = 256
MAX_THREADS = 8                # default thread cap (CHAMELEON_VLM_THREADS overrides)

ROLE_ENUM = ["background", "decorative", "logo", "hero", "image", "headline", "subhead",
             "body", "cta", "badge", "fine_print", "panel", "shape"]


class Backend(Protocol):
    def complete(self, png: "bytes | list[bytes]", prompt: str, schema: dict[str, Any],
                 max_tokens: int, deadline: float) -> str: ...


_state: dict[str, Any] = {
    "backend": None,           # Backend once loaded (or injected)
    "load_error": None,        # str when loading failed
    "disabled": None,          # str reason the adapter switched itself off
    "timeouts": 0,
    "failures": 0,
    "load_s": None,            # seconds the model took to load
    "cache": {},               # key -> dict | None (None = known-bad answer)
    "cache_order": [],
    "last": {},                # timing of the last real call (for the bench)
    "calls": 0,
}


# ---------------------------------------------------------------------------
# configuration


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def switched_off() -> bool:
    """CHAMELEON_VLM=0: the hard off switch."""
    return not _env_flag("CHAMELEON_VLM", True)


def model_paths() -> tuple[Path, Path] | None:
    """(text model, mmproj) GGUF paths from the environment or models/."""
    m = os.environ.get("CHAMELEON_VLM_MODEL", "").strip()
    p = os.environ.get("CHAMELEON_VLM_MMPROJ", "").strip()
    if m and p:
        return Path(m), Path(p)
    root = Path(os.environ.get("CHAMELEON_VLM_DIR", "") or MODELS_DIR)
    if not root.is_dir():
        return None
    ggufs = sorted(root.glob("*.gguf"))
    texts = [g for g in ggufs if not g.name.lower().startswith("mmproj")]
    projs = [g for g in ggufs if g.name.lower().startswith("mmproj")]
    if m:
        texts = [Path(m)]
    if p:
        projs = [Path(p)]
    if not texts or not projs:
        return None
    text = texts[0]
    # prefer the mmproj that shares the model's size tag (e.g. "2B", "4B")
    tag = re.search(r"(\d+)B", text.name)
    for proj in projs:
        if tag and tag.group(0) in proj.name:
            return text, proj
    return text, projs[0]


def available() -> bool:
    """Weights are present and the switch is on (the model may not be loaded yet)."""
    if switched_off() or _state["disabled"]:
        return False
    if _state["backend"] is not None:
        return True
    if _state["load_error"]:
        return False
    paths = model_paths()
    return bool(paths and paths[0].is_file() and paths[1].is_file())


def status() -> str:
    if switched_off():
        return "off: CHAMELEON_VLM=0"
    if _state["disabled"]:
        return "off: " + _state["disabled"]
    if _state["load_error"]:
        return "off: " + _state["load_error"]
    if _state["backend"] is not None:
        return "on (loaded in %.1fs)" % (_state["load_s"] or 0.0)
    paths = model_paths()
    if not paths or not paths[0].is_file() or not paths[1].is_file():
        return "off: no model (set CHAMELEON_VLM_MODEL/CHAMELEON_VLM_MMPROJ or put GGUFs in models/)"
    return "ready: " + paths[0].name


def stats() -> dict[str, Any]:
    """Load time and the timing of the last real (uncached) call."""
    return {"load_s": _state["load_s"], "calls": _state["calls"], "last": dict(_state["last"]),
            "timeouts": _state["timeouts"], "failures": _state["failures"], "status": status()}


def set_backend(backend: Optional[Backend]) -> None:
    """Inject a backend (tests, or a preloaded model). None forgets it."""
    _state["backend"] = backend
    _state["load_error"] = None


def _reset_for_tests() -> None:
    _state.update({"backend": None, "load_error": None, "disabled": None, "timeouts": 0,
                   "failures": 0, "load_s": None, "cache": {}, "cache_order": [], "last": {},
                   "calls": 0})


def _disable(reason: str) -> None:
    if not _state["disabled"]:
        _state["disabled"] = reason
        logger.warning("vlm switched off: %s", reason)


# ---------------------------------------------------------------------------
# llama.cpp backend


class LlamaCppBackend:
    """Qwen-VL style GGUF through llama-cpp-python's generic mtmd handler."""

    def __init__(self, model_path: Path, mmproj_path: Path, n_threads: int | None = None,
                 n_ctx: int = DEFAULT_N_CTX, seed: int = DEFAULT_SEED, use_gpu: bool = False,
                 chat_template: str | None = None):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler

        self.seed = seed
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.n_threads = n_threads or _default_threads()
        template = chat_template or os.environ.get("CHAMELEON_VLM_CHAT_TEMPLATE") or None

        class _Handler(MTMDChatHandler):
            # GGUFs without tokenizer.chat_template (e.g. moondream2) need an
            # explicit Jinja template; one from the environment wins
            def _get_chat_template(self, llama_model):
                if template:
                    return template
                return super()._get_chat_template(llama_model)

        handler = _Handler(clip_model_path=str(mmproj_path), verbose=False, use_gpu=use_gpu)
        self.llm = Llama(
            model_path=str(model_path),
            chat_handler=handler,
            n_ctx=n_ctx,
            n_batch=min(512, n_ctx),
            n_threads=self.n_threads,
            n_threads_batch=self.n_threads,
            n_gpu_layers=0 if not use_gpu else -1,
            seed=seed,
            logits_all=False,
            verbose=False,
        )
        self.last: dict[str, Any] = {}

    def complete(self, png: bytes | list[bytes], prompt: str, schema: dict[str, Any],
                 max_tokens: int, deadline: float) -> str:
        import base64

        pngs = png if isinstance(png, (list, tuple)) else [png]
        parts: list[dict[str, Any]] = []
        for i, data in enumerate(pngs):
            url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
            if len(pngs) > 1:
                parts.append({"type": "text", "text": "Image %d:" % (i + 1)})
            parts.append({"type": "image_url", "image_url": {"url": url}})
        parts.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": "You are a precise graphic-design analyst. Answer with JSON only."},
            {"role": "user", "content": parts},
        ]
        t0 = time.perf_counter()
        pieces: list[str] = []
        n_out = 0
        first = None
        stopped = False
        stream = self.llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repeat_penalty=1.0,
            seed=self.seed,
            max_tokens=max_tokens,
            stream=True,
        )
        try:
            for chunk in stream:
                now = time.perf_counter()
                if first is None:
                    first = now
                delta = chunk["choices"][0].get("delta", {}) or {}
                piece = delta.get("content")
                if piece:
                    pieces.append(piece)
                    n_out += 1
                if now > deadline:
                    stopped = True
                    break
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        end = time.perf_counter()
        self.last = {
            "prefill_s": (first or end) - t0,
            "gen_s": end - (first or end),
            "gen_tokens": n_out,
            "n_past": int(self.llm.n_tokens),      # positions, not tokens: M-RoPE counts an image as its grid height
            "stopped_at_deadline": stopped,
        }
        return "".join(pieces)


def _default_threads() -> int:
    v = os.environ.get("CHAMELEON_VLM_THREADS", "").strip()
    if v.isdigit() and int(v) > 0:
        return int(v)
    # llama.cpp scales up to the performance cores; the efficiency cores of an
    # Apple M-series chip slow prefill down (8 threads beat 11 on an M3 Pro)
    return max(1, min(MAX_THREADS, os.cpu_count() or 1))


def load(force: bool = False) -> Backend | None:
    """Load the configured model (once). None when off, missing or broken."""
    if switched_off() or _state["disabled"]:
        return None
    if _state["backend"] is not None and not force:
        return _state["backend"]
    if _state["load_error"] and not force:
        return None
    paths = model_paths()
    if not paths or not paths[0].is_file() or not paths[1].is_file():
        _state["load_error"] = "no model files"
        return None
    t0 = time.perf_counter()
    try:
        n_ctx = int(os.environ.get("CHAMELEON_VLM_CTX", "") or DEFAULT_N_CTX)
        seed = int(os.environ.get("CHAMELEON_VLM_SEED", "") or DEFAULT_SEED)
        use_gpu = _env_flag("CHAMELEON_VLM_GPU", False)
        backend = LlamaCppBackend(paths[0], paths[1], n_ctx=n_ctx, seed=seed, use_gpu=use_gpu)
    except Exception as exc:  # pragma: no cover - needs a broken install
        _state["load_error"] = "load failed: %s" % exc
        logger.warning("vlm unavailable: %s", exc)
        return None
    _state["load_s"] = time.perf_counter() - t0
    _state["backend"] = backend
    _state["load_error"] = None
    logger.info("vlm loaded %s + %s in %.1fs", paths[0].name, paths[1].name, _state["load_s"])
    return backend


# ---------------------------------------------------------------------------
# images, cache, parsing


def prepare_image(image, max_side: int = MAX_SIDE) -> bytes:
    """RGB PNG bytes, long side <= max_side (never upscaled)."""
    from PIL import Image

    img = image.convert("RGB") if image.mode != "RGB" else image
    w, h = img.size
    s = min(1.0, max_side / float(max(w, h)))
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def _cache_key(pngs: list[bytes], prompt: str, schema: dict[str, Any], max_tokens: int) -> str:
    h = hashlib.sha1()
    for png in pngs:
        h.update(hashlib.sha1(png).digest())
    h.update(prompt.encode("utf-8"))
    h.update(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(str(max_tokens).encode())
    h.update(_backend_tag().encode())
    return h.hexdigest()


def _backend_tag() -> str:
    b = _state["backend"]
    if b is None:
        paths = model_paths()
        return paths[0].name if paths else "none"
    return getattr(getattr(b, "model_path", None), "name", None) or type(b).__name__


def _cache_dir() -> Path | None:
    d = os.environ.get("CHAMELEON_VLM_CACHE_DIR", "").strip()
    return Path(d) if d else None


def _cache_get(key: str) -> tuple[bool, dict[str, Any] | None]:
    if key in _state["cache"]:
        return True, _state["cache"][key]
    d = _cache_dir()
    if d:
        f = d / (key + ".json")
        if f.is_file():
            try:
                return True, json.loads(f.read_text())
            except Exception:
                pass
    return False, None


def _cache_put(key: str, value: dict[str, Any] | None) -> None:
    cache, order = _state["cache"], _state["cache_order"]
    if value is not None:
        value = json.loads(json.dumps(value))     # the caller keeps its own copy
    if key not in cache:
        order.append(key)
        while len(order) > CACHE_ENTRIES:
            cache.pop(order.pop(0), None)
    cache[key] = value
    d = _cache_dir()
    if d and value is not None:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / (key + ".json")).write_text(json.dumps(value))
        except Exception:
            pass


def parse_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in `text`, or None. Tolerates chatter and code
    fences around it; never raises."""
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
        start = s.find("{", start + 1)
    return None


def validate(obj: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Problems with `obj` against a small JSON-schema subset (type, enum,
    const, properties/required/additionalProperties, items, minItems/maxItems,
    minimum/maximum, minLength/maxLength, pattern, anyOf). Empty = valid."""
    errs: list[str] = []
    if not isinstance(schema, dict):
        return errs
    if "anyOf" in schema:
        if not any(not validate(obj, s, path) for s in schema["anyOf"]):
            errs.append("%s: matches none of anyOf" % path)
        return errs
    if "const" in schema and obj != schema["const"]:
        errs.append("%s: expected %r" % (path, schema["const"]))
    if "enum" in schema and obj not in schema["enum"]:
        errs.append("%s: %r not in enum" % (path, obj))
    t = schema.get("type")
    types = t if isinstance(t, list) else ([t] if t else [])
    if types:
        ok = False
        for ty in types:
            if ty == "object" and isinstance(obj, dict):
                ok = True
            elif ty == "array" and isinstance(obj, list):
                ok = True
            elif ty == "string" and isinstance(obj, str):
                ok = True
            elif ty == "integer" and isinstance(obj, int) and not isinstance(obj, bool):
                ok = True
            elif ty == "number" and isinstance(obj, (int, float)) and not isinstance(obj, bool):
                ok = True
            elif ty == "boolean" and isinstance(obj, bool):
                ok = True
            elif ty == "null" and obj is None:
                ok = True
        if not ok:
            errs.append("%s: expected %s, got %s" % (path, "/".join(types), type(obj).__name__))
            return errs
    if isinstance(obj, dict):
        props = schema.get("properties", {}) or {}
        for key in schema.get("required", []) or []:
            if key not in obj:
                errs.append("%s: missing %r" % (path, key))
        for key, val in obj.items():
            if key in props:
                errs += validate(val, props[key], "%s.%s" % (path, key))
            elif "patternProperties" in schema:
                for pat, sub in schema["patternProperties"].items():
                    if re.fullmatch(pat, key):
                        errs += validate(val, sub, "%s.%s" % (path, key))
                        break
                else:
                    if schema.get("additionalProperties") is False:
                        errs.append("%s: unexpected key %r" % (path, key))
            elif schema.get("additionalProperties") is False:
                errs.append("%s: unexpected key %r" % (path, key))
            elif isinstance(schema.get("additionalProperties"), dict):
                errs += validate(val, schema["additionalProperties"], "%s.%s" % (path, key))
    elif isinstance(obj, list):
        if "minItems" in schema and len(obj) < schema["minItems"]:
            errs.append("%s: fewer than %d items" % (path, schema["minItems"]))
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            errs.append("%s: more than %d items" % (path, schema["maxItems"]))
        items = schema.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(obj):
                errs += validate(v, items, "%s[%d]" % (path, i))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append("%s: %r < minimum %r" % (path, obj, schema["minimum"]))
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append("%s: %r > maximum %r" % (path, obj, schema["maximum"]))
    elif isinstance(obj, str):
        if "minLength" in schema and len(obj) < schema["minLength"]:
            errs.append("%s: shorter than %d" % (path, schema["minLength"]))
        if "maxLength" in schema and len(obj) > schema["maxLength"]:
            errs.append("%s: longer than %d" % (path, schema["maxLength"]))
        if "pattern" in schema and not re.search(schema["pattern"], obj):
            errs.append("%s: does not match %r" % (path, schema["pattern"]))
    return errs


# ---------------------------------------------------------------------------
# the call


def describe(image, prompt: str, schema: dict[str, Any], max_tokens: int = DEFAULT_MAX_TOKENS,
             timeout_s: float = DEFAULT_TIMEOUT_S, max_side: int = MAX_SIDE,
             use_cache: bool = True) -> dict[str, Any] | None:
    """Ask the model about `image` (a PIL image, or a list of them: the prompt
    may refer to "Image 1", "Image 2", ...) and return a dict valid against
    `schema`, or None (off, no model, timeout, failure, invalid output).
    Never raises."""
    try:
        if switched_off() or _state["disabled"]:
            return None
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        pngs = [prepare_image(im, max_side) for im in images]
        png = pngs[0] if len(pngs) == 1 else pngs
        key = _cache_key(pngs, prompt, schema, max_tokens)
        if use_cache:
            hit, value = _cache_get(key)
            if hit:
                _state["last"] = {"cached": True}
                return json.loads(json.dumps(value)) if value is not None else None
        backend = load()
        if backend is None:
            return None
        deadline = time.perf_counter() + float(timeout_s)
        t0 = time.perf_counter()
        try:
            text = backend.complete(png, prompt, schema, int(max_tokens), deadline)
        except Exception as exc:
            _state["failures"] += 1
            logger.warning("vlm call failed (%s)", exc)
            if _state["failures"] >= MAX_FAILURES:
                _disable("backend raised %d times" % _state["failures"])
            return None
        dt = time.perf_counter() - t0
        last = dict(getattr(backend, "last", {}) or {})
        last.update({"total_s": dt, "cached": False})
        _state["last"] = last
        _state["calls"] += 1
        if dt > timeout_s and not last.get("stopped_at_deadline"):
            # the uninterruptible part alone overshot the deadline
            last["stopped_at_deadline"] = True
        if last.get("stopped_at_deadline"):
            _state["timeouts"] += 1
            logger.warning("vlm call hit its %.1fs deadline (%.1fs)", timeout_s, dt)
            if _state["timeouts"] >= MAX_TIMEOUTS:
                _disable("%d calls timed out" % _state["timeouts"])
            # whatever came back before the deadline is usually truncated;
            # keep it only if it still parses and validates
            parsed = parse_json(text)
            if parsed is None or validate(parsed, schema):
                return None
            if use_cache:
                _cache_put(key, parsed)
            return parsed
        parsed = parse_json(text)
        if parsed is None:
            logger.info("vlm output is not JSON: %r", (text or "")[:200])
            if use_cache:
                _cache_put(key, None)
            return None
        problems = validate(parsed, schema)
        if problems:
            logger.info("vlm output fails schema: %s", "; ".join(problems[:5]))
            if use_cache:
                _cache_put(key, None)
            return None
        if use_cache:
            _cache_put(key, parsed)
        return parsed
    except Exception as exc:  # belt and braces: never into the engine
        logger.warning("vlm adapter error (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# smoke test: python -m engine.vlm [task.json ...]


def _smoke(paths: list[str]) -> int:
    """Render each task's source and ask the model one question. Prints the
    status, timings and the answers; exit code 1 when nothing answered."""
    import glob

    from engine import render as R

    files = [p for pat in (paths or ["input/*.json"]) for p in sorted(glob.glob(pat))]
    schema = {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["product_ad", "event_poster", "editorial", "quote", "other"]},
        "headline": {"type": "string", "maxLength": 120},
        "n_text_blocks": {"type": "integer", "minimum": 0, "maximum": 50}},
        "required": ["kind", "headline", "n_text_blocks"], "additionalProperties": False}
    prompt = ("Look at this design. Reply with JSON: kind (product_ad | event_poster | editorial | "
              "quote | other), headline (the largest text, verbatim), n_text_blocks (how many "
              "separate pieces of text).")
    print("vlm:", status(), "| render:", R.status())
    answered = 0
    for f in files:
        task = json.loads(Path(f).read_text())
        src = task.get("source", task)
        img = R.render_design(src, max_size=800)
        if img is None:
            print(f, "-> no render")
            continue
        t0 = time.perf_counter()
        out = describe(img, prompt, schema, max_tokens=96, timeout_s=120)
        dt = time.perf_counter() - t0
        print("%s -> %s  (%.1fs; %s)" % (Path(f).name, json.dumps(out), dt,
                                          json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                                      for k, v in _state["last"].items()})))
        answered += out is not None
    print("vlm:", status(), "| load %.1fs" % (_state["load_s"] or 0.0), "| answered %d/%d" % (answered, len(files)))
    return 0 if answered else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_smoke(sys.argv[1:]))
