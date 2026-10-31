import { useRef, useState } from "react";
import type { Design, Preset, Sample, Target } from "../types";
import { Preview } from "./Preview";
import { Icon } from "./ui";

type Props = {
  samples: Sample[];
  presets: Preset[];
  sourceId: string | null;
  source: Design | null;
  sourceRender: string | null;
  onPickSample: (s: Sample) => void;
  onUpload: (d: Design, name: string, target?: { width: number; height: number }) => void;
  targets: Target[];
  setTargets: (t: Target[]) => void;
  onGenerate: () => void;
  onCancel: () => void;
  busy: boolean;
  progress: number;
  error: string | null;
};

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

  const added = p.targets.filter((t) => t.id.startsWith("custom") || t.id.startsWith("task"));

  return (
    <aside className="flex shrink-0 flex-col border-b border-white/[0.07] lg:w-[360px] lg:border-b-0 lg:border-r">
      <div className="space-y-7 p-6 lg:flex-1 lg:overflow-y-auto">
        <section>
          <div className="label mb-3">Design</div>
          <div className="grid grid-cols-3 gap-2">
            {p.samples.map((s, i) => (
              <button key={s.id} disabled={p.busy} onClick={() => p.onPickSample(s)} className={`option justify-center py-2 ${s.id === p.sourceId ? "option-on" : ""}`}>
                Sample {i + 1}
              </button>
            ))}
          </div>
          <div
            className="mt-3 rounded-lg border border-white/[0.07] bg-white/[0.02] p-4"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f && !p.busy) readFile(f);
            }}
          >
            {p.source && (
              <div className="flex justify-center">
                <Preview key={p.sourceRender ?? "none"} src={p.sourceRender} width={p.source.canvas.width} height={p.source.canvas.height} maxHeight={200} className="w-full" />
              </div>
            )}
            <div className="mt-3 flex items-center justify-between text-xs">
              <span className="font-mono text-zinc-500">{p.source && `${p.source.canvas.width}×${p.source.canvas.height}`}</span>
              <button className="inline-flex items-center gap-1.5 text-zinc-300 hover:text-white disabled:opacity-40" disabled={p.busy} onClick={() => fileRef.current?.click()}>
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
            {uploadErr && <div className="mt-2 text-xs text-rose-300">{uploadErr}</div>}
          </div>
        </section>

        <section>
          <div className="label mb-3">Sizes</div>
          <div className="space-y-4">
            {Object.entries(groups).map(([g, items]) => (
              <div key={g}>
                <div className="mb-1.5 text-xs text-zinc-500">{g}</div>
                <div className="flex flex-wrap gap-1.5">
                  {items.map((x) => (
                    <button key={x.id} onClick={() => toggle(x)} title={`${x.width}×${x.height}`} className={`option ${selected.has(x.id) ? "option-on" : ""}`}>
                      {x.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            <div>
              <div className="mb-1.5 text-xs text-zinc-500">Custom</div>
              <div className="flex items-center gap-1.5">
                <input aria-label="Width" value={cw} onChange={(e) => setCw(e.target.value.replace(/\D/g, ""))} placeholder="Width" className="field" />
                <span className="text-zinc-600">×</span>
                <input aria-label="Height" value={ch} onChange={(e) => setCh(e.target.value.replace(/\D/g, ""))} onKeyDown={(e) => e.key === "Enter" && addCustom()} placeholder="Height" className="field" />
                <button className="option !p-2" onClick={addCustom} aria-label="Add custom size">
                  <Icon name="plus" className="h-3.5 w-3.5" />
                </button>
              </div>
              {added.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {added.map((t) => (
                    <span key={t.id} className="option option-on">
                      <span className="font-mono">{t.width}×{t.height}</span>
                      <button onClick={() => p.setTargets(p.targets.filter((x) => x.id !== t.id))} aria-label={`Remove ${t.width}×${t.height}`} className="opacity-70 hover:opacity-100">
                        <Icon name="x" className="h-3 w-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      </div>

      <div className="border-t border-white/[0.07] p-6">
        {p.error && <div role="alert" className="mb-3 text-xs leading-relaxed text-rose-300">{p.error}</div>}
        {p.busy ? (
          <div role="status" className="space-y-3">
            <div className="flex items-center justify-between text-sm text-zinc-300">
              <span>Generating… {Math.round(p.progress * 100)}%</span>
              <button className="text-xs text-zinc-400 hover:text-white" onClick={p.onCancel}>Cancel</button>
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-white/[0.08]">
              <div className="h-full bg-emerald-300 transition-all" style={{ width: `${Math.max(4, p.progress * 100)}%` }} />
            </div>
          </div>
        ) : (
          <button className="btn-primary w-full" disabled={!p.source || !p.targets.length} onClick={p.onGenerate}>
            Generate {p.targets.length > 0 ? `${p.targets.length} size${p.targets.length > 1 ? "s" : ""}` : ""}
          </button>
        )}
      </div>
    </aside>
  );
}
