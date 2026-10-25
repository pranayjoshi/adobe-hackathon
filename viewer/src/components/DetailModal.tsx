import { useEffect, useState } from "react";
import { api, download, ratioLabel } from "../api";
import type { Candidate, Design, JudgeResult, TargetResult } from "../types";
import { Preview } from "./Preview";
import { Bar, Icon, ScoreRing, Toggle, roleColor } from "./ui";

type Props = {
  result: TargetResult;
  index: number;
  source: Design;
  name: string;
  roleOf: (id: string) => string | undefined;
  labelOf: (id: string) => string | undefined;
  onClose: () => void;
  onIndex: (i: number) => void;
};

export function DetailModal({ result, index, source, name, roleOf, labelOf, onClose, onIndex }: Props) {
  const c: Candidate = result.candidates[index];
  const [overlay, setOverlay] = useState(false);
  const [view, setView] = useState<"result" | "baseline">("result");
  const [copied, setCopied] = useState(false);
  const [judging, setJudging] = useState(false);
  const [verdict, setVerdict] = useState<JudgeResult | null>(null);
  const [judgeError, setJudgeError] = useState("");
  useEffect(() => { setVerdict(null); setJudgeError(""); }, [index]);
  const askJudge = async () => {
    setJudging(true); setJudgeError("");
    try { setVerdict(await api.judge(source, c.design)); } catch (e) { setJudgeError(String(e instanceof Error ? e.message : e)); } finally { setJudging(false); }
  };
  const { width, height } = c.design.canvas;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") onIndex(Math.min(result.candidates.length - 1, index + 1));
      if (e.key === "ArrowLeft") onIndex(Math.max(0, index - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, result, onClose, onIndex]);

  const srcById = Object.fromEntries(source.elements.map((e) => [e.id, e]));
  const showBaseline = view === "baseline";
  const issueIcon = { error: "x", warn: "warn", info: "info" } as const;
  const issueColor = { error: "text-rose-300", warn: "text-amber-300", info: "text-sky-300" } as const;

  return (
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/70 p-0 backdrop-blur-sm sm:p-6" onClick={onClose}>
      <div className="panel flex w-full max-w-[1400px] flex-col overflow-hidden !bg-ink-900 animate-rise lg:flex-row" onClick={(e) => e.stopPropagation()}>
        {/* Stage */}
        <div className="relative flex min-h-[320px] flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-3 border-b border-white/[0.06] px-5 py-3">
            <div className="flex rounded-lg border border-white/[0.07] bg-white/[0.02] p-0.5 text-xs">
              {(["result", "baseline"] as const).map((v) => (
                <button key={v} onClick={() => setView(v)} className={`rounded-md px-2.5 py-1 transition ${view === v ? "bg-white/10 text-white" : "text-zinc-400 hover:text-zinc-200"}`}>
                  {v === "result" ? "Chameleon" : "Naive stretch"}
                </button>
              ))}
            </div>
            <Toggle on={overlay && !showBaseline} onChange={setOverlay} label="Group overlay" />
            <div className="ml-auto flex items-center gap-1">
              {result.candidates.map((x, i) => (
                <button
                  key={x.template}
                  onClick={() => onIndex(i)}
                  className={`rounded-md px-2 py-1 text-xs transition ${i === index ? "bg-emerald-300/15 text-emerald-200" : "text-zinc-400 hover:bg-white/5"}`}
                >
                  {i + 1}. {x.name.split("·")[0].trim()}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-1 items-center justify-center overflow-auto bg-[radial-gradient(circle_at_center,rgba(255,255,255,.035),transparent_70%)] p-6 sm:p-10">
            <Preview
              key={(showBaseline ? result.baseline.render : c.render) ?? ""}
              src={showBaseline ? result.baseline.render : c.render}
              width={width}
              height={height}
              overlay={c.overlay}
              roleOf={roleOf}
              labelOf={labelOf}
              showOverlay={overlay && !showBaseline}
              maxHeight={Math.min(640, window.innerHeight - 220)}
              className="w-full"
            />
          </div>
          {showBaseline && (
            <div className="absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full border border-amber-300/20 bg-black/70 px-3 py-1.5 text-xs text-amber-200 backdrop-blur">
              The starter-kit baseline: every element stretched by the same x/y factors.
            </div>
          )}
        </div>

        {/* Inspector */}
        <aside className="flex w-full flex-col border-t border-white/[0.06] lg:w-[400px] lg:border-l lg:border-t-0">
          <div className="flex items-start gap-3 border-b border-white/[0.06] p-5">
            <ScoreRing value={c.score.total} size={56} stroke={5} />
            <div className="min-w-0 flex-1">
              <div className="text-[11px] uppercase tracking-wider text-zinc-500">
                {result.target.label ?? "Target"} · <span className="font-mono">{width}×{height}</span> · {ratioLabel(width, height)}
              </div>
              <h2 className="mt-0.5 font-display text-2xl leading-tight text-white">{c.name}</h2>
              <p className="mt-1 text-xs leading-relaxed text-zinc-400">{c.description}</p>
            </div>
            <button onClick={onClose} className="rounded-lg p-1.5 text-zinc-400 hover:bg-white/5 hover:text-white" aria-label="Close">
              <Icon name="x" />
            </button>
          </div>

          <div className="flex-1 space-y-6 overflow-y-auto p-5">
            <section className="space-y-2">
              <div className="label">Quality estimate</div>
              {c.quality?.passed === false && <p className="my-2 text-xs text-amber-300">Needs review: {c.quality.defect_count} layout issues. Preservation alone is not a quality pass.</p>}
              <Bar label="Readability" value={c.score.parts.readability} />
              <Bar label="Hierarchy" value={c.score.parts.hierarchy} />
              <Bar label="Composition" value={c.score.parts.composition} />
              <Bar label="Relative order" value={c.score.parts.fidelity} />
            </section>

            {c.rubric && (
              <section className="space-y-2">
                <div className="label">Design rubric</div>
                <div className="flex flex-wrap gap-1.5 text-[11px]">
                  {([['Tier 0 · rules', c.rubric.tier0.passed], ['Tier 1 · structure', c.rubric.tier1.passed], ['Tier 2 · no defects', !!c.rubric.tier2?.passed]] as const).map(([label, ok]) => (
                    <span key={label} className={`rounded-full px-2 py-0.5 ${ok ? 'bg-emerald-400/15 text-emerald-300' : 'bg-rose-400/15 text-rose-300'}`}>{ok ? '✓' : '✕'} {label}</span>
                  ))}
                  {c.rubric.tier2 && <span className="ml-auto font-mono text-zinc-300">overall {c.rubric.tier2.overall}</span>}
                </div>
                {c.rubric.tier2 && Object.entries(c.rubric.tier2.scores).map(([k, v]) => <Bar key={k} label={k[0].toUpperCase() + k.slice(1)} value={v} />)}
                {[...c.rubric.tier0.failures, ...c.rubric.tier1.failures, ...(c.rubric.tier2?.defects ?? []), ...c.rubric.tier1.warnings, ...(c.rubric.tier2?.warnings ?? [])].map((i, n) => (
                  <div key={i.id + n} className={`flex gap-2 text-xs leading-relaxed ${i.severity === 'defect' ? 'text-rose-300' : 'text-amber-300'}`}>
                    <span className="font-mono text-[10px] text-zinc-500">{i.id}</span>
                    <span className="text-zinc-300">{i.message}{i.ids.length ? ` (${i.ids.slice(0, 4).join(', ')}${i.ids.length > 4 ? '…' : ''})` : ''}</span>
                  </div>
                ))}
              </section>
            )}

            <section className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="label">AI judge (Tier 3)</div>
                <button className="btn-ghost ml-auto !py-1 text-xs" disabled={judging} onClick={askJudge}>{judging ? "Judging…" :verdict ? "Ask again" : "Ask AI judge"}</button>
              </div>
              {judgeError && <p className="text-xs text-rose-300">{judgeError}</p>}
              {verdict && (
                <>
                  <div className="font-mono text-xs text-zinc-300">overall {verdict.overall}/10 · {verdict.backend}</div>
                  {Object.entries(verdict.scores).map(([k, v]) => (
                    <div key={k} title={verdict.reasons[k]}><Bar label={k[0].toUpperCase() + k.slice(1)} value={v * 10} /><p className="mb-1 text-[11px] leading-snug text-zinc-500">{verdict.reasons[k]}</p></div>
                  ))}
                  {verdict.defects.map((d) => <p key={d} className="text-xs text-amber-300">• {d}</p>)}
                  <p className="text-[11px] text-zinc-500">A small local model: treat its verdict as a second opinion, not ground truth.</p>
                </>
              )}
            </section>

            <section>
              <div className="label mb-2">Notes for this size</div>
              {c.score.issues.length === 0 ? (
                <div className="flex items-center gap-2 text-xs text-emerald-300/80">
                  <Icon name="check" className="h-3.5 w-3.5" /> Everything fits comfortably.
                </div>
              ) : (
                <ul className="space-y-1.5">
                  {c.score.issues.map((i) => (
                    <li key={i.message} className={`flex gap-2 text-xs leading-relaxed ${issueColor[i.severity]}`}>
                      <Icon name={issueIcon[i.severity]} className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                      <span className="text-zinc-300">{i.message}</span>
                    </li>
                  ))}
                </ul>
              )}
              {c.render_warnings?.map((w) => (
                <div key={w} className="mt-1.5 flex gap-2 text-xs text-zinc-400">
                  <Icon name="info" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-sky-300" /> {w}
                </div>
              ))}
            </section>

            <section>
              <div className="label mb-2">Preservation checks</div>
              <ul className="divide-y divide-white/[0.05] rounded-xl border border-white/[0.06]">
                {c.checks.map((k) => (
                  <li key={k.name} className="flex items-center gap-2.5 px-3 py-2 text-xs">
                    <span className={`grid h-4 w-4 place-items-center rounded-full ${k.passed ? "bg-emerald-400/15 text-emerald-300" : "bg-rose-400/15 text-rose-300"}`}>
                      <Icon name={k.passed ? "check" : "x"} className="h-2.5 w-2.5" />
                    </span>
                    <span className="text-zinc-300">{k.name}</span>
                    <span className="ml-auto truncate pl-2 text-right font-mono text-[10px] text-zinc-500">{k.detail}</span>
                  </li>
                ))}
              </ul>
            </section>

            <section>
              <div className="label mb-2">Elements</div>
              <div className="overflow-hidden rounded-xl border border-white/[0.06]">
                <table className="w-full text-[11px]">
                  <thead className="bg-white/[0.03] text-zinc-500">
                    <tr>
                      <th className="px-3 py-1.5 text-left font-medium">id</th>
                      <th className="px-2 py-1.5 text-left font-medium">role</th>
                      <th className="px-3 py-1.5 text-right font-medium">size · type</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {c.design.elements.map((e) => {
                      const s = srcById[e.id];
                      const role = roleOf(e.id);
                      return (
                        <tr key={e.id} className="hover:bg-white/[0.02]">
                          <td className="max-w-[120px] truncate px-3 py-1.5 font-mono text-zinc-300">{e.id}</td>
                          <td className="px-2 py-1.5">
                            <span className="inline-flex items-center gap-1.5 text-zinc-400">
                              <span className="h-1.5 w-1.5 rounded-full" style={{ background: roleColor(role) }} />
                              {role}
                            </span>
                          </td>
                          <td className="px-3 py-1.5 text-right font-mono text-zinc-500">
                            {Math.round(e.width)}×{Math.round(e.height)}
                            {e.type === "text" && s && (
                              <span className="ml-2 text-zinc-400">
                                {s.font_size}→<span className="text-white">{e.font_size}</span>px
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          </div>

          <div className="flex gap-2 border-t border-white/[0.06] p-4">
            <button className="btn-primary flex-1" onClick={() => download(`${name}_${width}x${height}_${c.template}.json`, c.design)}>
              <Icon name="download" /> Download JSON
            </button>
            <button
              className="btn-ghost"
              onClick={async () => {
                await navigator.clipboard.writeText(JSON.stringify(c.design, null, 2));
                setCopied(true);
                setTimeout(() => setCopied(false), 1400);
              }}
            >
              <Icon name={copied ? "check" : "copy"} /> {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}
