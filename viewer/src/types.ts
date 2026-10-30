export type Element = {
  id: string;
  type: "text" | "image" | "shape";
  x: number;
  y: number;
  width: number;
  height: number;
  font_size?: number;
  content?: string;
  [k: string]: unknown;
};

export type Design = { canvas: { width: number; height: number }; elements: Element[] };

export type Sample = {
  id: string;
  title: string;
  notes: string;
  source: Design;
  target_canvas?: { width: number; height: number };
};

export type Preset = { id: string; group: string; label: string; width: number; height: number };

export type Box = { x: number; y: number; w: number; h: number };

export type BlockInfo = {
  id: string;
  role: string;
  label: string;
  kind: string;
  members: string[];
  box: Box;
  importance: number;
  children?: BlockInfo[];
};

export type GroupMode = 'rigid' | 'reflowable' | 'support';
export type AnalysisOverrides = {roles:Record<string,string>;group_modes:Record<string,GroupMode>};
export type AnalysisGraph = {
  version:string;
  nodes:Record<string,{type:string;role:string;role_confidence:number;role_source:string}>;
  group_policies:{group_id:string;member_ids:string[];mode:GroupMode;authority:string;kind?:string;editable_modes?:GroupMode[]}[];
};

export type Analysis = {
  graph?:AnalysisGraph|null;
  canvas: { width: number; height: number };
  roles: Record<string, string>;
  blocks: BlockInfo[];
  stickers: BlockInfo[];
  accents: BlockInfo[];
  background: string[];
  decorations: string[];
  notes: string[];
};

export type Check = { name: string; passed: boolean; detail: string; severity: string };
export type Issue = { severity: "error" | "warn" | "info"; message: string; ids: string[] };

export type Score = {
  total: number;
  parts: { readability: number; hierarchy: number; composition: number; fidelity: number };
  issues: Issue[];
  stats: { coverage: number; min_font: number | null };
};

export type RubricIssue = { id: string; severity: string; message: string; ids: string[] };
export type Rubric = {
  gate: boolean;
  passed: boolean;
  tier0: { passed: boolean; failures: RubricIssue[] };
  tier1: { passed: boolean; failures: RubricIssue[]; warnings: RubricIssue[] };
  tier2: { passed: boolean; overall: number; scores: Record<string, number>; defects: RubricIssue[]; warnings: RubricIssue[]; stats: Record<string, unknown> } | null;
};

export type Candidate = {
  revision_hash?: string;
  verification?: {verified:boolean;resource_hash?:string;profile?:string};
  controller?: {mode:string;stop_reason:string;events:ControllerEvent[];usage:Record<string,number>;incumbent_revision?:string;active_revision?:string};
  rubric?: Rubric;
  quality?: {passed:boolean;defect_count:number;defects:Record<string,unknown[]>};
  id?: string;
  diagnostics?: {overflow:string[];small_text:string[];collisions:unknown[];contrast:Record<string,number>;contrast_regressions:string[];render_mode:string};
  trace?: unknown[];
  template: string;
  appeal?: number;            // learned appeal score (0-1) when appeal selection ranked it
  preferred_learned?: boolean;
  name: string;
  description: string;
  design: Design;
  checks: Check[];
  valid: boolean;
  score: Score;
  overlay: (Box & { id: string })[];
  render: string | null;
  render_warnings: string[];
  preview_verified?: boolean;
};

export type TargetResult = {
  target: { width: number; height: number; label?: string | null; id?: string | null };
  candidates: Candidate[];
  baseline: { design: Design; render: string | null; checks: Check[] };
  elapsed_ms: number;
  status?: string;
};

export type GenerateResponse = { run_id?: string; analysis: Analysis; results: TargetResult[]; elapsed_ms: number };
export type ControllerMode = 'legacy_batch' | 'single' | 'single_restart' | 'small_beam';
export type ControllerEvent = {event_id?:number;sequence?:number;type:string;stage?:string;operation?:string;reason?:string;stop_reason?:string;revision_hash?:string;accepted?:boolean;verified?:boolean;target_index?:number};
export type ActivePreview = {render:string;revision_hash:string;target_index:number;width:number;height:number;operation?:string};
export type JobUpdate = {id:string;state:string;progress:number;stage?:string;result:GenerateResponse|null;error?:string;events:ControllerEvent[];active_preview?:ActivePreview};

export type Target = { id: string; label: string; width: number; height: number };
export type SavedGenerateRequest = {source:Design;analysis_overrides?:AnalysisOverrides;targets?:{width:number;height:number;id?:string|null;label?:string|null}[];options?:Record<string,boolean|string|number>;emphasis?:string;locks?:Record<string,Record<string,number>>;all_candidates?:boolean};

export type JudgeResult = { scores: Record<string, number>; reasons: Record<string, string>; defects: string[]; overall: number; samples: number; backend: string };
