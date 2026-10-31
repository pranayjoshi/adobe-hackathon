import { useState } from "react";
import { download } from "../api";
import type { Design, GenerateResponse, Target, TargetResult } from "../types";
import { Preview } from "./Preview";
import { Icon } from "./ui";

type Props = {
  resp: GenerateResponse | null;
  busy: boolean;
  targets: Target[];
  source: Design | null;
  sourceRender: string | null;
  name: string;
};

export function Results({ resp, busy, targets, source, sourceRender, name }: Props) {
  if (busy) {
    return (
      <div className="mx-auto max-w-4xl space-y-10 p-8">
        {targets.map((t) => (
          <div key={t.id}>
            <div className="mb-4 text-sm text-zinc-500">{t.label} <span className="ml-1 font-mono text-xs">{t.width}×{t.height}</span></div>
            <div className="mx-auto rounded-lg skeleton" style={{ aspectRatio: `${t.width}/${t.height}`, maxHeight: 460, maxWidth: (460 * t.width) / t.height }} />
          </div>
        ))}
      </div>
    );
  }
  if (!resp) {
    return (
      <div className="grid h-full min-h-[280px] place-items-center p-8 text-center">
        <div>
          <p className="font-display text-3xl text-white">Same elements, new canvas.</p>
          <p className="mt-2 text-sm text-zinc-500">Pick the sizes you need, then generate.</p>
        </div>
      </div>
    );
  }
  return (
    <div className="mx-auto max-w-4xl divide-y divide-white/[0.07] px-8">
      {resp.results.map((r, i) => (
        <SizeResult key={`${resp.run_id}-${i}`} result={r} source={source} sourceRender={sourceRender} name={name} />
      ))}
    </div>
  );
}

type View = "result" | "original" | "stretch";
const VIEWS: [View, string][] = [["result", "Result"], ["original", "Original"], ["stretch", "Plain stretch"]];

function SizeResult({ result, source, sourceRender, name }: { result: TargetResult; source: Design | null; sourceRender: string | null; name: string }) {
  const [index, setIndex] = useState(0);
  const [view, setView] = useState<View>("result");
  const { width, height } = result.target;
  const c = result.candidates[index];
  const title = (
    <h2 className="text-base font-medium text-white">
      {result.target.label ?? "Custom"} <span className="ml-1.5 font-mono text-xs font-normal text-zinc-500">{width}×{height}</span>
    </h2>
  );
  if (!c) {
    return (
      <section className="py-8">
        {title}
        <p role="alert" className="mt-3 text-sm text-amber-200">No layout was produced for this size.</p>
      </section>
    );
  }
  const shown = view === "original" && source
    ? { src: sourceRender, width: source.canvas.width, height: source.canvas.height }
    : view === "stretch" ? { src: result.baseline.render, width, height } : { src: c.render, width, height };
  const failed = c.checks.filter((k) => !k.passed && k.severity === "error");
  const fileName = `${name}_${width}x${height}${index ? `_option${index + 1}` : ""}`;

  return (
    <section className="py-8">
      <div className="mb-5 flex flex-wrap items-center gap-3">
        {title}
        <div role="group" aria-label="Compare" className="ml-auto flex rounded-lg border border-white/[0.08] p-0.5 text-xs">
          {VIEWS.map(([v, label]) => (
            <button key={v} aria-pressed={view === v} onClick={() => setView(v)} className={`rounded-md px-2.5 py-1 transition ${view === v ? "bg-white/10 text-white" : "text-zinc-400 hover:text-zinc-200"}`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex justify-center">
        <Preview key={shown.src ?? ""} src={shown.src} width={shown.width} height={shown.height} maxHeight={460} className="w-full" />
      </div>

      {result.candidates.length > 1 && (
        <div className="mt-5 flex flex-wrap items-center gap-2">
          {result.candidates.map((x, i) => (
            <button
              key={x.template}
              onClick={() => { setIndex(i); setView("result"); }}
              aria-label={i === 0 ? "Top pick" : `Option ${i + 1}`}
              aria-pressed={i === index}
              className={`rounded-md p-1 ring-1 transition ${i === index ? "ring-emerald-300" : "ring-white/10 hover:ring-white/30"}`}
            >
              <div style={{ width: Math.min(160, (56 * width) / height) }}>
                <Preview src={x.render} width={width} height={height} className="w-full !rounded !ring-0" />
              </div>
            </button>
          ))}
          <span className="ml-1 text-xs text-zinc-500">{index === 0 ? "Top pick" : `Option ${index + 1}`}</span>
        </div>
      )}

      {result.status === "unverified_missing_asset" && (
        <p role="status" className="mt-4 text-xs text-zinc-500">An image this design refers to wasn't supplied, so the preview shows a placeholder in its place.</p>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <details className="min-w-0 flex-1 text-xs text-zinc-400">
          <summary className="flex cursor-pointer list-none items-center gap-1.5">
            <Icon name={failed.length ? "warn" : "check"} className={`h-3.5 w-3.5 ${failed.length ? "text-amber-300" : "text-emerald-300"}`} />
            {failed.length ? `${failed.length} of ${c.checks.length} rule checks failed` : `All ${c.checks.length} rule checks passed`}
          </summary>
          <ul className="mt-3 space-y-1.5">
            {c.checks.map((k) => (
              <li key={k.name} className="flex items-start gap-2">
                <Icon name={k.passed ? "check" : "x"} className={`mt-0.5 h-3 w-3 shrink-0 ${k.passed ? "text-emerald-300" : "text-rose-300"}`} />
                <span>{k.name}{!k.passed && k.detail ? `: ${k.detail}` : ""}</span>
              </li>
            ))}
          </ul>
        </details>
        <div className="flex self-start gap-2">
          {c.render && (
            <a className="btn-ghost" href={c.render} download={`${fileName}.png`}>
              <Icon name="download" className="h-3.5 w-3.5" /> PNG
            </a>
          )}
          <button className="btn-ghost" onClick={() => download(`${fileName}.json`, c.design)}>
            <Icon name="download" className="h-3.5 w-3.5" /> JSON
          </button>
        </div>
      </div>
    </section>
  );
}
