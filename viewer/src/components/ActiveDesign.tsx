import {useRef,useState} from 'react';
import {api,download} from '../api';
import type {Candidate,Element} from '../types';

type Patch=Record<string,Record<string,number>>;
type Props={candidate:Candidate;runId?:string;targetIndex:number;locked:string[];
  onCandidate:(candidate:Candidate)=>void;onLock:(element:Element)=>void};

/** The SVG's screen matrix is the sole canvas/view transform, including zoom
 * and letterboxing. Its image remains the official render while edits are
 * pending; text boxes change constraints and never stretch rendered glyphs. */
export function ActiveDesign({candidate:c,runId,targetIndex,locked,onCandidate,onLock}:Props){
  const [selected,setSelected]=useState(''),[patches,setPatches]=useState<Patch>({});
  const [zoom,setZoom]=useState(1),[busy,setBusy]=useState(false),[notice,setNotice]=useState('');
  const [past,setPast]=useState<string[]>([]),[future,setFuture]=useState<string[]>([]);
  const drag=useRef<{id:string;x:number;y:number;original:Element;resize:boolean}|null>(null);
  const svg=useRef<SVGSVGElement>(null);
  const {width,height}=c.design.canvas;
  const elements=c.design.elements.map(e=>({...e,...patches[e.id]}));
  const active=elements.find(e=>e.id===selected);
  const dirty=Object.keys(patches).length>0;
  const coordinate=(event:React.PointerEvent<SVGElement>)=>{
    const inverse=svg.current?.getScreenCTM()?.inverse();
    return inverse?new DOMPoint(event.clientX,event.clientY).matrixTransform(inverse):null;
  };
  const patch=(id:string,change:Record<string,number>)=>setPatches(old=>({...old,[id]:{...old[id],...change}}));
  const apply=async()=>{
    if(!runId||!c.revision_hash)return;
    setBusy(true);setNotice('Validating and rendering correction…');
    try{
      const result=await api.edit(runId,targetIndex,c.revision_hash,patches);
      setPast([...past,c.revision_hash]);setFuture([]);setPatches({});
      onCandidate(result.candidate);setNotice('Correction committed and saved as a human demonstration.');
    }catch(error){setNotice(String(error));}finally{setBusy(false);}
  };
  const restore=async(direction:'undo'|'redo')=>{
    if(direction==='undo'&&dirty){setPatches({});setNotice('Uncommitted geometry reset.');return;}
    if(!runId||!c.revision_hash)return;
    const list=direction==='undo'?past:future;const hash=list[list.length-1];if(!hash)return;
    setBusy(true);
    try{
      const result=await api.restore(runId,targetIndex,c.revision_hash,hash);
      if(direction==='undo'){setPast(past.slice(0,-1));setFuture([...future,c.revision_hash]);}
      else{setFuture(future.slice(0,-1));setPast([...past,c.revision_hash]);}
      setPatches({});onCandidate(result.candidate);setNotice(direction==='undo'?'Previous verified revision restored.':'Correction restored.');
    }catch(error){setNotice(String(error));}finally{setBusy(false);}
  };
  const vote=async(verdict:'like'|'dislike')=>{
    if(!runId||!c.revision_hash||dirty)return;
    try{await api.feedback(runId,targetIndex,c.revision_hash,verdict,undefined,c.verification?.resource_hash);setNotice(`Feedback saved for this exact design: ${verdict}.`);}
    catch(error){setNotice(String(error));}
  };
  return <div className="panel overflow-hidden">
    <div className="flex flex-wrap items-center gap-3 border-b border-white/10 p-4">
      <h3 className="font-semibold">Active design</h3>
      <span className="text-xs text-zinc-400">{c.controller?.stop_reason?.replaceAll('_',' ')??'Verified result'}</span>
      <label className="ml-auto text-xs">Zoom <select aria-label="Canvas zoom" className="bg-zinc-900 p-1" value={zoom} onChange={event=>setZoom(Number(event.target.value))}>{[.5,1,1.5,2,3].map(value=><option key={value} value={value}>{value*100}%</option>)}</select></label>
      <button className="chip" onClick={()=>download(`design-${width}x${height}.json`,c.design)}>Download JSON</button>
    </div>
    <div className="max-h-[65vh] overflow-auto bg-black/20 p-4">
      <svg ref={svg} viewBox={`0 0 ${width} ${height}`} width={`${zoom*100}%`} style={{minWidth:240,aspectRatio:`${width}/${height}`,maxWidth:'none',maxHeight:height/width>2?700*zoom:undefined}}
        role="group" aria-label="Official design preview with editable geometry" tabIndex={0}
        onKeyDown={event=>{
          if(busy){event.preventDefault();return;}
          if(event.key==='Escape'){setPatches({});drag.current=null;return;}
          if((event.ctrlKey||event.metaKey)&&event.key==='z'){event.preventDefault();void restore(event.shiftKey?'redo':'undo');return;}
          if(!active||locked.includes(active.id)||busy)return;
          const delta=event.shiftKey?10:1;const arrows:Record<string,[number,number]>={ArrowLeft:[-delta,0],ArrowRight:[delta,0],ArrowUp:[0,-delta],ArrowDown:[0,delta]};
          if(arrows[event.key]){event.preventDefault();const [dx,dy]=arrows[event.key];patch(active.id,{x:active.x+dx,y:active.y+dy});}
        }}
        onPointerMove={event=>{
          if(!drag.current)return;const point=coordinate(event);if(!point)return;
          const {id,x,y,original,resize}=drag.current;
          if(resize){const w=Math.max(1,original.width+point.x-x);const h=original.type==='image'?w*original.height/original.width:Math.max(1,original.height+point.y-y);patch(id,{width:w,height:h});}
          else patch(id,{x:original.x+point.x-x,y:original.y+point.y-y});
        }} onPointerUp={()=>{drag.current=null;}} onPointerCancel={()=>{drag.current=null;}}>
        {c.render&&<image href={c.render} width={width} height={height}/>}
        {!c.render&&<text x={width/2} y={height/2} textAnchor="middle">Official preview unavailable</text>}
        {elements.map(element=><rect key={element.id} x={element.x} y={element.y} width={element.width} height={element.height}
          fill={selected===element.id?'rgba(16,185,129,.07)':'transparent'} stroke={selected===element.id?'#34d399':'transparent'} strokeWidth={1.5} vectorEffect="non-scaling-stroke"
          style={{cursor:locked.includes(element.id)?'not-allowed':'move'}}
          onPointerDown={event=>{
            event.preventDefault();setSelected(element.id);const point=coordinate(event);
            if(!point||locked.includes(element.id)||busy)return;
            event.currentTarget.setPointerCapture(event.pointerId);
            drag.current={id:element.id,x:point.x,y:point.y,original:element,resize:false};
          }}><title>{element.content?.slice(0,80)||element.id}</title></rect>)}
        {active&&!locked.includes(active.id)&&<rect x={active.x+active.width-5} y={active.y+active.height-5} width={10} height={10} fill="#34d399" vectorEffect="non-scaling-stroke" style={{cursor:'nwse-resize'}}
          onPointerDown={event=>{const point=coordinate(event);if(!point||busy)return;event.currentTarget.setPointerCapture(event.pointerId);drag.current={id:active.id,x:point.x,y:point.y,original:active,resize:true};}}/>}
      </svg>
    </div>
    <div className="space-y-3 border-t border-white/10 p-4 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <label>Element <select aria-label="Edit element" className="max-w-64 bg-zinc-900 p-2" value={selected} onChange={event=>setSelected(event.target.value)}><option value="">Select an element</option>{elements.map(element=><option key={element.id} value={element.id}>{element.content?.slice(0,35)||element.id}</option>)}</select></label>
        {active&&<><button className="chip" onClick={()=>onLock(active)} disabled={locked.includes(active.id)}>Lock for next generation</button>{(['x','y','width','height',...(active.type==='text'?['font_size']:[])] as const).map(key=><label key={key}>{key}<input className="ml-1 w-20 bg-zinc-900 p-2" aria-label={`Element ${key}`} type="number" step={key==='font_size'?1:.1} value={Number(active[key]).toFixed(key==='font_size'?0:1)} disabled={busy||locked.includes(active.id)} onChange={event=>{const value=Number(event.target.value);if(Number.isFinite(value))patch(active.id,{[key]:value});}}/></label>)}</>}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button className="chip" disabled={!dirty||busy||!runId} onClick={apply}>Apply correction</button>
        <button className="chip" disabled={busy||(!dirty&&!past.length)} onClick={()=>restore('undo')}>Undo</button>
        <button className="chip" disabled={busy||!future.length||dirty} onClick={()=>restore('redo')}>Redo</button>
        <button className="chip" disabled={!dirty||busy} onClick={()=>setPatches({})}>Reset pending</button>
        <span className="ml-auto text-zinc-400">{dirty?'Pending box changes; apply to reflow text and render.':'Rate the saved rendered design'}</span>
        <button className="chip" disabled={!runId||dirty||busy||!c.render} onClick={()=>vote('like')}>Like</button>
        <button className="chip" disabled={!runId||dirty||busy||!c.render} onClick={()=>vote('dislike')}>Needs work</button>
      </div>
      <p role="status" className="text-zinc-400">{notice||'Select an element; drag, resize, or use arrow keys. Shift moves by ten pixels. Images keep their aspect ratio.'}</p>
      {c.controller&&<details><summary className="cursor-pointer text-emerald-300">Improvement history · {c.controller.events.length} events</summary><ol className="mt-3 max-h-64 space-y-2 overflow-auto">{c.controller.events.map((event,index)=><li key={index} className="border-l border-white/10 pl-3"><span className={event.type==='accepted'?'text-emerald-300':'text-zinc-400'}>{event.type}</span> · {event.operation||event.stage||event.stop_reason||event.reason}{event.verified?' · rendered and verified':''}</li>)}</ol></details>}
    </div>
  </div>;
}
