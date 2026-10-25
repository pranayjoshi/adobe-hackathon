import {capabilities,preflight} from './capabilities.js';
let source,candidates=[],proxy,selectRevision=0,backendProfile=null,busy=false;
const $=id=>document.getElementById(id),say=s=>$('status').textContent=s;
const base=()=>$('base').value.replace(/\/$/,'');
async function readResponse(response){const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
function compatibility(candidate){
  if(!candidate)return {supported:false,errors:[]};
  const report=preflight(candidate.design);
  if(!candidate.valid)report.errors.push({id:null,property:'validity',message:'The engine marked this candidate invalid.'});
  if(!backendProfile||backendProfile.adapter_version!==capabilities().adapter_version)report.errors.push({id:null,property:'profile',message:'Engine editable-export profile unavailable or incompatible.'});
  report.supported=report.errors.length===0;return report;
}
function selected(){
  const candidate=candidates[Number($('candidate').value)||0];
  const report=compatibility(candidate);$('apply').disabled=busy||!proxy||!report.supported;
  if(candidate){$('preview').src=base()+candidate.render;
    $('compatibility').textContent=report.supported?'Editable subset supported. Live Express rendering parity pending.':report.errors.map(e=>(e.id??'Design')+': '+e.message).join('\n');}
  return candidate;
}
if(window.parent!==window){try{const {default:sdk,RuntimeType}=await import('https://new.express.adobe.com/static/add-on-sdk/sdk.js');await sdk.ready;proxy=await sdk.instance.runtime.apiProxy(RuntimeType.documentSandbox);await proxy.capabilities();}catch(error){say('Express connection unavailable: '+error.message);}}
$('file').onchange=async()=>{try{const task=JSON.parse(await $('file').files[0].text());source=task.source??task;candidates=[];selectRevision++;$('apply').disabled=true;say(source.elements.length+' elements loaded.');}catch(error){say(error.message);}};
$('candidate').onchange=()=>{selectRevision++;selected();};
$('base').onchange=()=>{selectRevision++;backendProfile=null;selected();};
$('width').onchange=$('height').onchange=()=>{selectRevision++;};
$('generate').onclick=async()=>{
  if(busy)return;
  if(!source)return say('Upload source JSON first.');
  busy=true;const requestRevision=++selectRevision;selected();
  try{
    say('Generating…');
    backendProfile=await readResponse(await fetch(base()+'/api/express/capabilities'));
    const result=await readResponse(await fetch(base()+'/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source,targets:[{width:Number($('width').value),height:Number($('height').value)}],top_k:3})}));
    if(requestRevision!==selectRevision)throw new Error('Source or target changed during generation. Generate the current design again.');
    candidates=result.results[0].candidates;
    $('candidate').replaceChildren(...candidates.map((candidate,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=candidate.name;return option;}));
    say('Generation complete. Compatibility is checked separately before application.');
  }catch(error){say(error.message);}finally{busy=false;selected();}
};
$('apply').onclick=async()=>{
  const candidate=selected(),revision=selectRevision;
  if(!proxy||!candidate||!compatibility(candidate).supported)return;
  busy=true;selected();
  try{
    const report=await proxy.preflight(candidate.design);
    if(!report.supported)throw new Error(report.errors.map(e=>e.message).join('\n'));
    const data=await readResponse(await fetch(base()+'/api/express/assets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({design:candidate.design})}));
    if(revision!==selectRevision)throw new Error('Selection changed while resources were prepared. Apply the current revision again.');
    const result=await proxy.apply({design:candidate.design,assets:data.assets});
    window.chameleonLastApplication=result;
    say(result.applied+' editable elements added. '+result.warnings.join(' '));
  }catch(error){say(error.message);}finally{busy=false;selected();}
};
