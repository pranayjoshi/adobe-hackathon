/** Which part of the engine produced a candidate, from its template tag.
 *  learned_s0_<style>: constrained solver over the model's layout; learned_strip: reading-order
 *  strip composer; learned_<k>[_raw|_bound]: a model sample; refined_ / solver_ prefixes are later
 *  engine passes (hill-climb refinement, the engine's own placement solver). */
export type Method = { label: string; detail: string; learned: boolean };

export function methodOf(template: string): Method {
  let t = template;
  const passes: string[] = [];
  for (;;) {
    if (t.startsWith("refined_")) { passes.push("refined"); t = t.slice(8); continue; }
    if (t.startsWith("solver_")) { passes.push("engine solver"); t = t.slice(7); continue; }
    break;
  }
  let label: string;
  let learned = true;
  const s0 = t.match(/^learned_s0_(\w+)$/);
  const sample = t.match(/^learned_(\d+)(_bound)?(_raw)?$/);
  if (s0) label = `Solver · ${s0[1]}`;
  else if (t === "learned_strip") label = "Strip composer";
  else if (sample) label = `Model sample ${sample[1]}${sample[2] ? " · copy-bound" : ""}${sample[3] ? " · raw" : ""}`;
  else { label = "Template"; learned = false; }
  const detail = [learned ? "designer-trained layout" : t, ...passes.reverse().map((p) => `+ ${p}`)].join(" ");
  return { label, detail, learned };
}
