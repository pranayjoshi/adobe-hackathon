export type Element = {
  id: string;
  type: "text" | "image" | "shape";
  x: number;
  y: number;
  width: number;
  height: number;
  [k: string]: unknown;
};

export type Design = { canvas: { width: number; height: number }; elements: Element[] };

export type Sample = {
  id: string;
  title: string;
  source: Design;
  target_canvas?: { width: number; height: number };
};

export type Preset = { id: string; group: string; label: string; width: number; height: number };

export type Check = { name: string; passed: boolean; detail: string; severity: string };

export type Candidate = {
  template: string;
  design: Design;
  checks: Check[];
  render: string | null;
};

export type TargetResult = {
  target: { width: number; height: number; label?: string | null };
  candidates: Candidate[];
  baseline: { render: string | null };
  status?: string;
};

export type GenerateResponse = { run_id?: string; results: TargetResult[] };
export type JobUpdate = { id: string; state: string; progress: number; result: GenerateResponse | null; error?: string; events: { event_id?: number }[] };

export type Target = { id: string; label: string; width: number; height: number };
