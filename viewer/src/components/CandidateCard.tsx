import { download } from "../api";
import { methodOf } from "../method";
import type { Candidate } from "../types";
import { Preview } from "./Preview";
import { Bar, Icon, ScoreRing } from "./ui";

type Props = {
  c: Candidate;
  rank: number;
  name: string;
  showOverlay: boolean;
  roleOf: (id: string) => string | undefined;
  labelOf: (id: string) => string | undefined;
  onOpen: () => void;
  wide: boolean;
};

export function CandidateCard({ c, rank, name, showOverlay, roleOf, labelOf, onOpen, wide }: Props) {
  const { width, height } = c.design.canvas;
  const issues = c.score.issues;
  const worst = issues.find((i) => i.severity === "error") ?? issues.find((i) => i.severity === "warn");
  const method = methodOf(c.template);
  return (
    <article
      className={`group panel animate-rise flex flex-col overflow-hidden transition hover:border-white/15 hover:bg-ink-850/90 ${wide ? "xl:flex-row" : ""}`}
      style={{ animationDelay: `${rank * 70}ms` }}
    >
      <button onClick={onOpen} className={`relative flex items-center justify-center bg-black/20 ${wide ? "p-4 xl:min-w-0 xl:flex-1" : "p-5"}`} style={{ minHeight: wide ? 0 : 280 }}>
        <Preview
          key={c.render ?? ""}
          src={c.render}
          width={width}
          height={height}
          overlay={c.overlay}
          roleOf={roleOf}
          labelOf={labelOf}
          showOverlay={showOverlay}
          maxHeight={wide ? 220 : 340}
          className="w-full"
        />
        <span className="absolute right-3 top-3 rounded-md bg-black/60 p-1.5 text-zinc-300 opacity-0 backdrop-blur transition group-hover:opacity-100">
          <Icon name="expand" className="h-3.5 w-3.5" />
        </span>
        {rank === 0 && (
          <span className={`absolute left-3 top-3 rounded-full ${c.quality?.passed === false ? "bg-amber-300" : "bg-emerald-300"} px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-ink-950 shadow-lg`}>
            {c.quality?.passed === false ? "Needs review" : "Top ranked"}
          </span>
        )}
      </button>
      <div className={`flex flex-1 flex-col gap-3 border-t border-white/[0.06] p-4 ${wide ? "xl:w-[340px] xl:flex-none xl:border-l xl:border-t-0" : ""}`}>
        {c.preview_verified === false && <p className="text-xs text-amber-300" role="status">Unverified preview. A required original image is missing.</p>}
        {c.quality?.passed === false && <p className="text-xs text-amber-300" role="status">No clean result for this option: {c.quality.defect_count} layout issues. Preservation checks do not certify visual quality.</p>}
        {c.diagnostics && <details className="text-xs text-zinc-400"><summary>Placement diagnostics and repair trace</summary><pre className="max-h-56 overflow-auto whitespace-pre-wrap">{JSON.stringify({diagnostics:c.diagnostics,trace:c.trace},null,2)}</pre></details>}
        <div className="flex items-start gap-3">
          <ScoreRing value={c.score.total} />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h3 className="truncate text-sm font-semibold text-white">{c.name}</h3>
              {c.valid ? (
                <span className="inline-flex items-center gap-0.5 rounded-full bg-emerald-400/10 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300">
                  <Icon name="check" className="h-3 w-3" /> preservation passed
                </span>
              ) : (
                <span className="rounded-full bg-rose-400/10 px-1.5 py-0.5 text-[10px] font-medium text-rose-300">invalid</span>
              )}
            </div>
            <p className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px]" title={method.detail + (typeof c.appeal === "number" ? ` · appeal score ${c.appeal.toFixed(2)}` : "")}>
              <span className={`rounded-full px-1.5 py-0.5 font-medium ${method.learned ? "bg-sky-400/10 text-sky-300" : "bg-white/[0.06] text-zinc-400"}`}>{method.label}</span>
              {method.detail.includes("+") && <span className="text-zinc-500">{method.detail.slice(method.detail.indexOf("+"))}</span>}
              {c.preferred_learned && <span className="text-emerald-300">{typeof c.appeal === "number" ? "picked by appeal" : "preferred layout"}</span>}
            </p>
            <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-zinc-500">{c.description}</p>
          </div>
        </div>
        <div className="space-y-1.5">
          <Bar label="Readability" value={c.score.parts.readability} />
          <Bar label="Hierarchy" value={c.score.parts.hierarchy} />
          <Bar label="Composition" value={c.score.parts.composition} />
          <Bar label="Relative order" value={c.score.parts.fidelity} />
        </div>
        <div className="mt-auto flex items-center justify-between gap-2 pt-1">
          <div className="min-w-0 flex-1 text-[11px]">
            {worst ? (
              <span className={`flex items-center gap-1.5 ${worst.severity === "error" ? "text-rose-300" : "text-amber-300/90"}`}>
                <Icon name="warn" className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{worst.message}</span>
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-emerald-300/80">
                <Icon name="check" className="h-3.5 w-3.5" /> No layout issues found
              </span>
            )}
          </div>
          <button
            className="btn-ghost !px-2 !py-1 !text-xs"
            onClick={() => download(`${name}_${width}x${height}_${c.template}.json`, c.design)}
            title="Download JSON"
          >
            <Icon name="download" className="h-3.5 w-3.5" /> JSON
          </button>
        </div>
      </div>
    </article>
  );
}
