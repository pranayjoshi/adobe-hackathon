import type { Design, JobUpdate, Preset, Sample, Target } from "./types";

async function req<T>(path: string, body?: unknown, signal?:AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* keep status text */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  samples: () => req<Sample[]>("/api/samples"),
  presets: () => req<Preset[]>("/api/presets"),
  preview: (design: Design, signal?: AbortSignal) => req<{ render: string | null }>("/api/analyze", { design, group_policies: "legacy" }, signal),
  generate: async (source: Design, targets: Target[], onProgress?: (id: string, progress: number) => void, signal?: AbortSignal) => {
    const started = await req<{ id: string }>("/api/jobs", { source, targets, top_k: 3, options: { engine: "final" }, idempotency_key: crypto.randomUUID() }, signal);
    let after = -1;
    for (;;) {
      const job = await req<JobUpdate>(`/api/jobs/${started.id}?after_event=${after}`, undefined, signal);
      after = Math.max(after, ...job.events.map((event) => event.event_id ?? -1));
      onProgress?.(started.id, job.progress);
      if (job.state === "done" && job.result) return job.result;
      if (job.state === "cancelled" && job.result?.results.length) return job.result;
      if (job.state === "cancelled") throw new Error("Generation cancelled.");
      if (job.state === "failed" || job.state === "interrupted") throw new Error(job.error ?? "No design was produced.");
      await new Promise<void>((resolve, reject) => {
        const finish = () => { signal?.removeEventListener("abort", abort); resolve(); };
        const timer = setTimeout(finish, 350);
        const abort = () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); };
        signal?.addEventListener("abort", abort, { once: true });
        if (signal?.aborted) abort();
      });
    }
  },
  cancel: (id: string) => req(`/api/jobs/${id}/cancel`, {}),
};

export function download(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
