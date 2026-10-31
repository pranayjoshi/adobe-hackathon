import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Results } from "./components/Results";
import { Sidebar } from "./components/Sidebar";
import { Logo } from "./components/ui";
import type { Design, GenerateResponse, Preset, Sample, Target } from "./types";

export default function App() {
  const generationAbort = useRef<AbortController | null>(null);
  const currentJob = useRef("");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState("design");
  const [source, setSource] = useState<Design | null>(null);
  const [sourceRender, setSourceRender] = useState<string | null>(null);
  const [targets, setTargets] = useState<Target[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [resp, setResp] = useState<GenerateResponse | null>(null);

  const loadSource = useCallback((d: Design, id: string | null, name: string, task?: { width: number; height: number }) => {
    setSource(d);
    setSourceId(id);
    setSourceName(name);
    setResp(null);
    setSourceRender(null);
    setError(null);
    if (task) {
      setTargets((prev) => {
        const keep = prev.filter((t) => !t.id.startsWith("task"));
        const exists = keep.some((t) => t.width === task.width && t.height === task.height);
        return exists ? keep : [{ id: `task-${task.width}x${task.height}`, label: "Requested size", ...task }, ...keep];
      });
    }
  }, []);

  useEffect(() => {
    if (!source) return;
    const abort = new AbortController();
    api.preview(source, abort.signal).then((result) => setSourceRender(result.render)).catch((e) => {
      if (!abort.signal.aborted) setError(String(e));
    });
    return () => abort.abort();
  }, [source]);

  useEffect(() => {
    api.presets().then(setPresets).catch(() => {});
    api.samples().then((s) => {
      setSamples(s);
      const first = s.find((x) => x.id.includes("level2")) ?? s[0];
      if (first) loadSource(first.source, first.id, first.id.replace("chameleon_sample_", ""), first.target_canvas);
    }).catch((e) => setError(`Engine service unreachable. Start it with uvicorn. (${e.message})`));
    return () => generationAbort.current?.abort();
  }, [loadSource]);

  const generate = async () => {
    if (!source) return;
    const abort = new AbortController();
    generationAbort.current = abort;
    setBusy(true);
    setProgress(0);
    setError(null);
    try {
      setResp(await api.generate(source, targets, (id, p) => { currentJob.current = id; setProgress(p); }, abort.signal));
    } catch (e) {
      if (!abort.signal.aborted) setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      currentJob.current = "";
    }
  };

  const cancel = () => {
    if (currentJob.current) api.cancel(currentJob.current).catch((e) => setError(String(e)));
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto lg:overflow-hidden">
      <header className="flex shrink-0 items-center border-b border-white/[0.07] px-6 py-4">
        <Logo />
      </header>
      <div className="flex flex-col lg:min-h-0 lg:flex-1 lg:flex-row">
        <Sidebar
          samples={samples}
          presets={presets}
          sourceId={sourceId}
          source={source}
          sourceRender={sourceRender}
          onPickSample={(s) => loadSource(s.source, s.id, s.id.replace("chameleon_sample_", ""), s.target_canvas)}
          onUpload={(d, name, task) => loadSource(d, null, name, task)}
          targets={targets}
          setTargets={setTargets}
          onGenerate={generate}
          onCancel={cancel}
          busy={busy}
          progress={progress}
          error={error}
        />
        <main className="min-w-0 flex-1 lg:overflow-y-auto">
          <Results resp={resp} busy={busy} targets={targets} source={source} sourceRender={sourceRender} name={sourceName} />
        </main>
      </div>
    </div>
  );
}
