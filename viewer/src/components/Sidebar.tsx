import { useRef, useState } from "react";
import { ratioLabel } from "../api";
import type { Analysis, AnalysisOverrides, Design, Preset, Sample, Target } from "../types";
import { IntentInspector } from "./IntentInspector";
import { Preview } from "./Preview";
import { Icon, roleColor } from "./ui";

type Props = {
  onBrowseCrello: () => void;
  crelloReference?: string;
  sourceName: string;
  samples: Sample[];
  presets: Preset[];
  sourceId: string | null;
  source: Design | null;
  sourceRender: string | null;
  analysis: Analysis | null;
  analysisOverrides: AnalysisOverrides;
  onAnalysisOverrides: (value:AnalysisOverrides)=>void;
  semanticLayout: boolean;
  intentPending: boolean;
  onPickSample: (s: Sample) => void;
  onUpload: (d: Design, name: string, target?: { width: number; height: number }) => void;
  targets: Target[];
  setTargets: (t: Target[]) => void;
  emphasis: string;
  setEmphasis: (e: string) => void;
  onGenerate: () => void;
  busy: boolean;
  error: string | null;
};

const EMPHASIS = [
  { id: "balanced", label: "Balanced" },
  { id: "headline", label: "Headline" },
  { id: "product", label: "Product" },
];

export function Sidebar(p: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [cw, setCw] = useState("");
  const [ch, setCh] = useState("");
  const [uploadErr, setUploadErr] = useState<string | null>(null);

  const groups = p.presets.reduce<Record<string, Preset[]>>((acc, x) => {
    (acc[x.group] ||= []).push(x);
    return acc;
  }, {});
  const selected = new Set(p.targets.map((t) => t.id));
  const toggle = (x: Preset) =>
    p.setTargets(selected.has(x.id) ? p.targets.filter((t) => t.id !== x.id) : [...p.targets, { id: x.id, label: x.label, width: x.width, height: x.height }]);

  const addCustom = () => {
    const w = parseInt(cw, 10);
    const h = parseInt(ch, 10);
    if (!(w > 0 && h > 0 && w <= 8000 && h <= 8000)) return;
    const id = `custom-${w}x${h}`;
    if (!selected.has(id)) p.setTargets([...p.targets, { id, label: "Custom", width: w, height: h }]);
    setCw("");
    setCh("");
  };

  const readFile = async (f: File) => {
    setUploadErr(null);
    try {
      const j = JSON.parse(await f.text());
      const design: Design = j.source ?? j;
      if (!design?.canvas || !Array.isArray(design.elements)) throw new Error("Expected {canvas, elements} or a task with 'source'.");
      p.onUpload(design, f.name.replace(/\.json$/i, ""), j.target_canvas);
    } catch (e) {
      setUploadErr(e instanceof Error ? e.message : String(e));
    }
  };

  // Count layout units (a button is one unit, not two elements).
  const roleCounts = p.analysis
    ? [
        ...p.analysis.background.map(() => "background"),
        ...p.analysis.decorations.map(() => "decoration"),
        ...[...p.analysis.blocks, ...p.analysis.stickers, ...p.analysis.accents].map((b) => b.role),
      ].reduce<Record<string, number>>((a, r) => ((a[r] = (a[r] || 0) + 1), a), {})
    : {};

  return (
    <aside className="flex h-full w-full flex-col lg:w-[360px] lg:shrink-0">
      <div className="flex-1 space-y-6 overflow-y-auto px-5 pb-6 pt-2">
        {/* Source */}
        <section>
          <div className="mb-2.5 flex items-center justify-between">
            <span className="label">1 · Source design</span>
            <button className="text-xs text-zinc-400 hover:text-emerald-300 inline-flex items-center gap-1" onClick={() => fileRef.current?.click()}>
              <Icon name="upload" className="h-3.5 w-3.5" /> Upload JSON
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) readFile(f);
                e.target.value = "";
              }}
            />
          </div>
          <button className="btn-ghost mb-3 w-full" disabled={p.busy} onClick={p.onBrowseCrello}>Browse Crello designs</button>
          <div className="grid grid-cols-3 gap-2">
            {p.samples.map((s, i) => {
              const active = s.id === p.sourceId;
              return (
                <button
                  key={s.id}
                  disabled={p.busy}
                  onClick={() => p.onPickSample(s)}
                  title={s.notes}
                  className={`group rounded-xl border px-2.5 py-2 text-left transition ${
                    active ? "border-emerald-300/50 bg-emerald-300/[0.07]" : "border-white/[0.07] bg-white/[0.02] hover:border-white/20"
                  }`}
                >
                  <div className={`text-[10px] font-semibold uppercase tracking-wider ${active ? "text-emerald-300" : "text-zinc-500"}`}>Level {i + 1}</div>
                  <div className="mt-0.5 truncate text-[13px] text-zinc-200">{s.title.split("—").pop()?.trim()}</div>
                  <div className="mt-0.5 font-mono text-[10px] text-zinc-500">{s.source.elements.length} elements</div>
                </button>
              );
            })}
          </div>
          <div
            className="mt-3 rounded-2xl border border-dashed border-white/10 p-3"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) readFile(f);
            }}
          >
            {p.source ? (
              <>
                <p className="mb-2 text-sm text-zinc-300">{p.sourceName}</p>
                {p.crelloReference && <details className="mb-3 text-xs text-zinc-400"><summary className="cursor-pointer">Compare dataset preview</summary><img src={p.crelloReference} alt="Crello dataset preview" className="mx-auto mt-2 max-h-52"/><p className="mt-2">Below: editable reconstruction. Text wrapping and effects can differ.</p></details>}
                <div className="flex justify-center">
                  <Preview key={p.sourceRender ?? "none"} src={p.sourceRender} width={p.source.canvas.width} height={p.source.canvas.height} maxHeight={220} className="w-full" />
                </div>
                <div className="mt-2.5 flex items-center justify-between text-xs">
                  <span className="font-mono text-zinc-400">
                    {p.source.canvas.width}×{p.source.canvas.height}
                    <span className="ml-1.5 text-zinc-600">{ratioLabel(p.source.canvas.width, p.source.canvas.height)}</span>
                  </span>
                  <span className="text-zinc-500">drop a JSON to replace</span>
                </div>
                {p.analysis && (
                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    {Object.entries(roleCounts).map(([r, n]) => (
                      <span key={r} className="chip !py-0.5 !text-[11px]">
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: roleColor(r) }} />
                        {r}
                        {n > 1 && <span className="text-zinc-500">×{n}</span>}
                      </span>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="py-8 text-center text-sm text-zinc-500">Pick a sample or drop a design JSON here</div>
            )}
            {uploadErr && <div className="mt-2 text-xs text-rose-300">{uploadErr}</div>}
          </div>
        </section>

        {p.source&&<IntentInspector source={p.source} analysis={p.analysis} value={p.analysisOverrides}
          onChange={p.onAnalysisOverrides} semantic={p.semanticLayout} pending={p.intentPending} busy={p.busy}/>}

        {/* Targets */}
        <section>
          <div className="mb-2.5 flex items-center justify-between">
            <span className="label">2 · Output sizes</span>
            {p.targets.length > 0 && (
              <button className="text-xs text-zinc-500 hover:text-zinc-300" onClick={() => p.setTargets([])}>
                Clear ({p.targets.length})
              </button>
            )}
          </div>
          <div className="space-y-3">
            {Object.entries(groups).map(([g, items]) => (
              <div key={g}>
                <div className="mb-1.5 text-[11px] text-zinc-500">{g}</div>
                <div className="flex flex-wrap gap-1.5">
                  {items.map((x) => {
                    const on = selected.has(x.id);
                    return (
                      <button
                        key={x.id}
                        onClick={() => toggle(x)}
                        className={`group flex items-center gap-2 rounded-lg border px-2 py-1.5 text-left text-xs transition ${
                          on ? "border-emerald-300/50 bg-emerald-300/10 text-emerald-100" : "border-white/[0.07] bg-white/[0.02] text-zinc-300 hover:border-white/20"
                        }`}
                      >
                        <RatioGlyph w={x.width} h={x.height} on={on} />
                        <span>
                          {x.label}
                          <span className={`ml-1.5 font-mono text-[10px] ${on ? "text-emerald-300/70" : "text-zinc-500"}`}>
                            {x.width}×{x.height}
                          </span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
            <div>
              <div className="mb-1.5 text-[11px] text-zinc-500">Custom</div>
              <div className="flex items-center gap-1.5">
                <input value={cw} onChange={(e) => setCw(e.target.value.replace(/\D/g, ""))} placeholder="Width" className="w-20 rounded-lg border border-white/10 bg-white/[0.03] px-2 py-1.5 font-mono text-xs outline-none focus:border-emerald-300/50" />
                <span className="text-zinc-600">×</span>
                <input value={ch} onChange={(e) => setCh(e.target.value.replace(/\D/g, ""))} onKeyDown={(e) => e.key === "Enter" && addCustom()} placeholder="Height" className="w-20 rounded-lg border border-white/10 bg-white/[0.03] px-2 py-1.5 font-mono text-xs outline-none focus:border-emerald-300/50" />
                <button className="btn-ghost !px-2 !py-1.5" onClick={addCustom} aria-label="Add custom size">
                  <Icon name="plus" className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {p.targets
                  .filter((t) => t.id.startsWith("custom") || t.id.startsWith("task"))
                  .map((t) => (
                    <span key={t.id} className="chip border-emerald-300/30 text-emerald-100">
                      {t.label} <span className="font-mono text-[10px] text-emerald-300/70">{t.width}×{t.height}</span>
                      <button onClick={() => p.setTargets(p.targets.filter((x) => x.id !== t.id))} className="text-zinc-400 hover:text-white">
                        <Icon name="x" className="h-3 w-3" />
                      </button>
                    </span>
                  ))}
              </div>
            </div>
          </div>
        </section>

        {/* Emphasis */}
        <section>
          <div className="label mb-2.5">3 · Emphasis</div>
          <div className="grid grid-cols-3 rounded-xl border border-white/[0.07] bg-white/[0.02] p-1">
            {EMPHASIS.map((e) => (
              <button
                key={e.id}
                onClick={() => p.setEmphasis(e.id)}
                className={`rounded-lg py-1.5 text-xs font-medium transition ${p.emphasis === e.id ? "bg-white/10 text-white shadow-inner" : "text-zinc-400 hover:text-zinc-200"}`}
              >
                {e.label}
              </button>
            ))}
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
            Nudges the solver to give the headline or product more room. Content, fonts and assets are never changed.
          </p>
        </section>
      </div>

      <div className="border-t border-white/[0.06] p-5">
        {p.error && <div className="mb-3 rounded-lg border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-xs text-rose-200">{p.error}</div>}
        <button className="btn-primary w-full !py-3" disabled={!p.source || !p.targets.length || p.busy} onClick={p.onGenerate}>
          {p.busy ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-ink-950/30 border-t-ink-950" /> Generating…
            </>
          ) : (
            <>
              <Icon name="spark" /> Generate {p.targets.length > 0 ? `${p.targets.length} size${p.targets.length > 1 ? "s" : ""}` : "layouts"}
            </>
          )}
        </button>
      </div>
    </aside>
  );
}

function RatioGlyph({ w, h, on }: { w: number; h: number; on: boolean }) {
  const k = 14 / Math.max(w, h);
  return (
    <span className="grid h-4 w-4 place-items-center">
      <span
        className={`block rounded-[2px] border ${on ? "border-emerald-300 bg-emerald-300/30" : "border-zinc-500"}`}
        style={{ width: Math.max(3, w * k), height: Math.max(3, h * k) }}
      />
    </span>
  );
}
