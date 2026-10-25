import { useEffect, useRef, useState } from 'react';
import { download } from '../api';
import type { Design } from '../types';

export type CrelloDesign = {
  id:string;source_id:string;title:string;format:string;category:string;width:number;height:number;
  elements:number;text_elements:number;image_elements:number;fonts:string[];missing_fonts:string[];
  status:string;reference:string;reconstruction:string;notes:string[];split:string;
};
type Collection={items:CrelloDesign[];total:number;collection_total:number;formats:string[];description:string;revision:string;dataset_url:string};
async function read<T>(url:string):Promise<T>{
  const r=await fetch(url);if(!r.ok){const e=await r.json();throw new Error(typeof e.detail==='string'?e.detail:e.detail?.message || 'Could not load this design');}return r.json();
}
export function CrelloBrowser({onClose,onSelect}:{onClose:()=>void;onSelect:(d:Design,item:CrelloDesign)=>void}){
  const selectionRequest=useRef(0);
  const scrollArea=useRef<HTMLDivElement>(null);
  const [q,setQ]=useState(''),[format,setFormat]=useState(''),[orientation,setOrientation]=useState(''),[status,setStatus]=useState(''),[offset,setOffset]=useState(0);
  const [data,setData]=useState<Collection|null>(null),[selected,setSelected]=useState<CrelloDesign|null>(null),[design,setDesign]=useState<Design|null>(null);
  const [loading,setLoading]=useState(false),[error,setError]=useState('');
  useEffect(()=>{let active=true;setLoading(true);setError('');const timer=setTimeout(()=>{
    const params=new URLSearchParams({q,format,orientation,status,offset:String(offset),limit:'18'});
    read<Collection>('/api/crello?'+params).then(d=>{if(active)setData(d);}).catch(e=>{if(active)setError(e.message);}).finally(()=>{if(active)setLoading(false);});
  },180);return()=>{active=false;clearTimeout(timer);};},[q,format,orientation,status,offset]);
  useEffect(()=>{const close=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose();};window.addEventListener('keydown',close);return()=>window.removeEventListener('keydown',close);},[onClose]);
  async function inspect(item:CrelloDesign){const request=++selectionRequest.current;setSelected(item);setDesign(null);setError('');scrollArea.current?.scrollTo({top:0});try{const r=await read<{source:Design}>('/api/crello/'+item.id);if(request===selectionRequest.current)setDesign(r.source);}catch(e){if(request===selectionRequest.current)setError(String(e));}}
  const field='rounded-lg border border-white/10 bg-zinc-900 px-3 py-2 text-sm';
  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-6" role="dialog" aria-modal="true" aria-label="Crello design browser">
    <div className="flex max-h-full w-full max-w-7xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-zinc-950 shadow-2xl">
      <header className="flex items-start justify-between gap-4 border-b border-white/10 p-5"><div><h2 className="text-2xl font-semibold">Crello designs</h2><p className="mt-1 text-sm text-zinc-400">{data?.collection_total ?? '…'} local designs · Original artwork and editable text · Validation collection</p></div><button className="btn-ghost" onClick={onClose}>Close</button></header>
      <div ref={scrollArea} className="flex min-h-0 flex-1 flex-col overflow-y-auto lg:flex-row">
        <section className={`min-w-0 flex-1 p-5 ${selected?'lg:max-w-[58%]':''}`}>
          <div className="flex flex-wrap gap-2"><input aria-label="Search Crello designs" className={field+' min-w-40 flex-1'} placeholder="Search titles, themes or fonts…" value={q} onChange={e=>{setQ(e.target.value);setOffset(0);}}/>
            <select aria-label="Crello format" className={field} value={format} onChange={e=>{setFormat(e.target.value);setOffset(0);}}><option value="">All formats</option>{data?.formats.map(f=><option key={f}>{f}</option>)}</select>
            <select aria-label="Crello orientation" className={field} value={orientation} onChange={e=>{setOrientation(e.target.value);setOffset(0);}}><option value="">All shapes</option><option value="square">Square</option><option value="portrait">Portrait</option><option value="landscape">Landscape</option></select>
            <select aria-label="Crello resource status" className={field} value={status} onChange={e=>{setStatus(e.target.value);setOffset(0);}}><option value="">All resource statuses</option><option value="ready">Resources ready</option><option value="needs-review">Needs review</option></select>
          </div>
          <p className="my-3 text-xs text-zinc-500" role="status">{loading?'Loading designs…':`${data?.total ?? 0} matching designs. Thumbnails show the dataset preview.`}</p>
          {!loading && !data?.items.length && <p className="py-8 text-zinc-400">{data?.collection_total?'No designs match these filters.':'No local Crello collection. See docs/crello-browser.md for the import step.'}</p>}
          <div className={`grid gap-3 ${selected?'grid-cols-2 xl:grid-cols-3':'grid-cols-2 sm:grid-cols-3 lg:grid-cols-6'}`}>
            {data?.items.map(item=><button key={item.id} onClick={()=>inspect(item)} aria-label={'Inspect '+item.title} className={`overflow-hidden rounded-xl border text-left transition hover:border-emerald-300/70 ${selected?.id===item.id?'border-emerald-300':'border-white/10'}`}><div className="flex h-40 items-center justify-center bg-zinc-900 p-2"><img loading="lazy" src={item.reference} alt={item.title} className="max-h-full max-w-full object-contain"/></div><div className="p-2.5"><div className="line-clamp-2 min-h-10 text-sm">{item.title}</div><p className="mt-1 text-[11px] text-zinc-400">{item.width} × {item.height} · {item.elements} layers</p><p className={`mt-1 text-[11px] ${item.status==='ready'?'text-emerald-300':'text-amber-300'}`}>{item.status==='ready'?'Resources ready':'Needs review'}</p></div></button>)}
          </div>
          <div className="mt-4 flex items-center justify-between"><button className="btn-ghost" disabled={offset===0 || loading} onClick={()=>setOffset(Math.max(0,offset-18))}>Previous</button><span className="text-xs text-zinc-400">{data?.total?`${offset+1}–${Math.min(offset+18,data.total)} of ${data.total}`:'0 results'}</span><button className="btn-ghost" disabled={!data || offset+18>=data.total || loading} onClick={()=>setOffset(offset+18)}>Next</button></div>
        </section>
        {selected && <section className="order-first min-w-0 flex-1 border-b border-white/10 p-5 lg:order-last lg:border-b-0 lg:border-l"><h3 className="text-lg font-semibold">{selected.title}</h3><p className="mt-1 text-xs text-zinc-400">{selected.format} · {selected.width} × {selected.height} · {selected.split}</p>
          <div className="mt-4 grid grid-cols-2 gap-3">{[[selected.reference,'Dataset preview'],[selected.reconstruction,'Editable reconstruction']].map(([url,title])=><figure key={url}><div className="flex h-52 items-center justify-center rounded-xl bg-zinc-900 p-2"><img src={url} alt={title} className="max-h-full max-w-full object-contain"/></div><figcaption className="mt-2 text-center text-xs text-zinc-400">{title}</figcaption></figure>)}</div>
          <p className="mt-4 text-sm text-zinc-300">{selected.text_elements} editable text layers · {selected.image_elements} original image layers</p><p className="mt-2 text-xs text-zinc-400">Fonts: {selected.fonts.join(', ') || 'No text'}</p>
          <p className="mt-2 text-xs text-zinc-400">{selected.status==='ready'?'Original assets and requested font faces are available.':'Some font faces or rendering details need review.'} Resource availability does not certify visual fidelity.</p>
          <ul className="mt-3 list-disc space-y-2 pl-4 text-xs leading-relaxed text-amber-100/70">{selected.notes.map((n,i)=><li key={i}>{n}</li>)}</ul>
          <p className="mt-4 text-[10px] text-zinc-500">Source ID: {selected.source_id} · Revision {data?.revision?.slice(0,12)}</p>
          <div className="mt-4 flex flex-wrap gap-2"><button disabled={!design || loading} className="btn-primary" onClick={()=>{if(design)onSelect(design,selected);}}>Use this design</button><button disabled={!design} className="btn-ghost" onClick={()=>download('crello-'+selected.source_id+'.json',{source:design,provenance:{dataset:'cyberagent/crello',revision:data?.revision,source_id:selected.source_id,split:selected.split}})}>Download design JSON</button></div>
        </section>}
      </div>
      {error && <p role="alert" className="px-5 py-3 text-sm text-rose-300">{error}</p>}
      <footer className="border-t border-white/10 px-5 py-3 text-[11px] text-zinc-500">Imported locally for testing · CDLA-Permissive-2.0 · <a className="underline" href={data?.dataset_url || 'https://huggingface.co/datasets/cyberagent/crello'} target="_blank" rel="noreferrer">Dataset and provenance</a></footer>
    </div>
  </div>;
}
