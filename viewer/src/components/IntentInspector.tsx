import type { Analysis, AnalysisOverrides, Design, GroupMode } from '../types';

const ROLES: Record<string, string[]> = {
  text: ['headline','subhead','body','legal','cta','badge','logo','unknown'],
  image: ['hero','image','logo','cta','badge','decoration','unknown'],
  shape: ['cta','badge','unknown'],
};
const MODE_LABEL: Record<GroupMode,string> = {
  rigid:'Keep relative arrangement', reflowable:'Allow measured reflow', support:'Keep copy on its panel',
};

type Props = {
  source: Design;
  analysis: Analysis | null;
  value: AnalysisOverrides;
  onChange: (value: AnalysisOverrides) => void;
  semantic: boolean;
  pending: boolean;
  busy: boolean;
};

export function IntentInspector({source,analysis,value,onChange,semantic,pending,busy}:Props) {
  const memberIds=new Set<string>();
  const visit=(blocks:Analysis['blocks'])=>blocks.forEach(b=>{b.members.forEach(id=>memberIds.add(id));if(b.children)visit(b.children);});
  if(analysis)visit([...analysis.blocks,...analysis.stickers,...analysis.accents]);
  const elements=source.elements.filter(e=>memberIds.has(e.id));
  const policies=analysis?.graph?.group_policies??[];
  const count=Object.keys(value.roles).length+Object.keys(value.group_modes).length;
  const updateRole=(id:string,role:string)=>{
    const roles={...value.roles};if(role)roles[id]=role;else delete roles[id];onChange({...value,roles});
  };
  const updateMode=(id:string,mode:string)=>{
    const group_modes={...value.group_modes};if(mode)group_modes[id]=mode as GroupMode;else delete group_modes[id];onChange({...value,group_modes});
  };
  return <section className="rounded-xl border border-white/10 bg-white/[0.02] p-3" aria-label="Design intent inspector">
    <details>
      <summary className="cursor-pointer text-sm font-medium text-zinc-200">Design intent {count>0&&<span className="text-xs text-emerald-300">· {count} correction{count===1?'':'s'}</span>}{pending&&<span className="ml-2 text-xs text-amber-200">Pending</span>}</summary>
      <p className="mt-2 text-xs leading-relaxed text-zinc-400">Correct an element’s purpose or a group’s behavior. Changes apply when you press Generate. Your source JSON stays unchanged.</p>
      {pending&&<p role="status" className="mt-2 text-xs text-amber-200">Intent changes are pending for the next generation.</p>}
      {!analysis&&<p className="mt-3 text-xs text-zinc-500">Reading source structure…</p>}
      {analysis&&<div className="mt-4 space-y-3">
        <h3 className="text-xs font-semibold text-zinc-300">Element purpose</h3>
        <div className="max-h-72 space-y-3 overflow-y-auto pr-1">
          {elements.map(e=>{
            const node=analysis.graph?.nodes[e.id];
            const detected=analysis.roles[e.id]??'unknown';
            const confidence=node?.role_confidence;
            return <label key={e.id} className="block space-y-1 text-xs">
              <span className="block truncate text-zinc-300" title={e.id}>{e.content?.slice(0,48)||e.id}</span>
              <span className="block text-[10px] text-zinc-500">{e.type} · {node?.role_source??'heuristic'}{typeof confidence==='number'&&Number.isFinite(confidence)?` · confidence ${confidence.toFixed(2)}`:''}</span>
              <select aria-label={`Purpose of ${e.id}`} disabled={busy} className="w-full rounded-lg border border-white/10 bg-zinc-900 p-2 text-zinc-200 disabled:opacity-50" value={value.roles[e.id]??''} onChange={event=>updateRole(e.id,event.target.value)}>
                <option value="">Detected: {detected}</option>
                {(ROLES[e.type]??[]).map(role=><option key={role} value={role}>{role}</option>)}
              </select>
            </label>;
          })}
          {elements.length===0&&<p className="text-xs text-zinc-500">No editable content roles in this analysis.</p>}
        </div>
        <p className="text-[10px] leading-relaxed text-zinc-500">Source backgrounds and decorations stay read-only. Choosing “Detected” removes your correction.</p>
        <h3 className="pt-2 text-xs font-semibold text-zinc-300">Group behavior</h3>
        {!semantic&&<p className="text-xs text-zinc-500">Enable Semantic layout to set group behavior. Saved group corrections are inactive while it is off.</p>}
        <div className="max-h-60 space-y-3 overflow-y-auto pr-1">
          {policies.map(policy=>{
            // Only the engine may declare which transitions an actual group supports.
            const allowed=policy.editable_modes??[policy.mode];
            const editable=semantic&&allowed.length>1;
            return <label key={policy.group_id} className="block space-y-1 text-xs">
              <span className="block truncate text-zinc-300" title={policy.group_id}>{policy.kind?.replaceAll('_',' ')??'group'} · {policy.member_ids.length} elements</span>
              <span className="block truncate text-[10px] text-zinc-500" title={policy.member_ids.join(', ')}>{policy.authority} · {policy.member_ids.join(', ')}</span>
              {editable?<select aria-label={`Behavior of ${policy.group_id}`} disabled={busy} className="w-full rounded-lg border border-white/10 bg-zinc-900 p-2 text-zinc-200 disabled:opacity-50" value={value.group_modes[policy.group_id]??''} onChange={event=>updateMode(policy.group_id,event.target.value)}>
                <option value="">Detected: {MODE_LABEL[policy.mode]}</option>
                {allowed.map(mode=><option key={mode} value={mode}>{MODE_LABEL[mode]}</option>)}
              </select>:<span className="block text-xs text-zinc-400">{MODE_LABEL[semantic?(value.group_modes[policy.group_id]??policy.mode):policy.mode]} · read-only</span>}
            </label>;
          })}
          {policies.length===0&&<p className="text-xs text-zinc-500">No group policies available in this analysis.</p>}
        </div>
        {Object.keys(value.group_modes).length>0&&!semantic&&<p className="text-xs text-amber-200">Group corrections will resume when Semantic layout is enabled.</p>}
        {count>0&&<button type="button" className="chip" disabled={busy} onClick={()=>onChange({roles:{},group_modes:{}})}>Reset intent corrections</button>}
        <p className="text-[10px] leading-relaxed text-zinc-500">Custom alignment guides and focal-point crop controls are not available in this document format.</p>
      </div>}
    </details>
  </section>;
}
