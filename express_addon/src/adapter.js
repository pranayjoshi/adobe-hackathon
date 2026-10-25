import {capabilities,fontKey,parseColor,preflight} from './capabilities.js';

export function createAdapter({editor,constants,fonts,decode=atob}) {
  let applicationSequence=0;
  async function prepare({design,assets={}}){
    const report=preflight(design);
    if(!report.supported)throw new Error(report.errors.map(e=>(e.id??'design')+'.'+e.property+': '+e.message).join('\n'));
    const images=new Map(),fontMap=new Map();
    for(const e of design.elements){
      if(e.type==='image'){
        if(typeof assets[e.id]!=='string'||!assets[e.id])throw new Error('Prepared original image unavailable: '+e.id);
        const bytes=Uint8Array.from(decode(assets[e.id]),c=>c.charCodeAt(0));
        if(!bytes.length)throw new Error('Empty image resource: '+e.id);
        images.set(e.id,await editor.loadBitmapImage(new Blob([bytes],{type:'image/png'})));
      }
      if(e.type==='text'){
        const requested=fontKey(e),font=await fonts.fromPostscriptName(requested);
        if(!font||font.availableForEditing===false)throw new Error('Editable font unavailable: '+requested);
        fontMap.set(e.id,font);
      }
    }
    return {images,fontMap,report};
  }
  async function apply(payload){
    // Freeze before awaiting: resource completion cannot apply a different edit.
    const design=JSON.parse(JSON.stringify(payload.design));
    const {images,fontMap,report}=await prepare({...payload,design});
    let result;
    // Async-loaded bitmap/font handles are applied in the SDK's synchronous
    // follow-up, which also makes edits visible to host save/Undo tracking.
    await editor.keepContentActiveDuringAsync(editor.context.insertionParent,async()=>{},()=>{
    let page;const mapping=[],observed=[];
    try{
      page=editor.documentRoot.pages.addPage({width:design.canvas.width,height:design.canvas.height});
      const board=page.artboards.first;
      for(const e of [...design.elements].sort((a,b)=>(a.z_index??0)-(b.z_index??0))){
        let node;
        if(e.type==='text'){
          node=editor.createText(e.content);
          node.layout={type:constants.TextLayout.autoHeight,width:e.width};
          node.fullContent.applyCharacterStyles({font:fontMap.get(e.id),fontSize:e.font_size*report.profile.units.font_px_to_points,color:parseColor(e.color)},{start:0,length:e.content.length});
          node.textAlignment=constants.TextAlignment[e.text_align??'left'];
        } else if(e.type==='image')node=editor.createImageContainer(images.get(e.id),{initialSize:{width:e.width,height:e.height}});
        else{
          node=e.shape_type==='ellipse'?editor.createEllipse():editor.createRectangle();
          node.width=e.width;node.height=e.height;node.fill=editor.makeColorFill(parseColor(e.fill_color));
        }
        node.opacity=e.opacity??1;
        board.children.append(node);
        node.setPositionInParent({x:e.x,y:e.y},node.topLeftLocal??{x:0,y:0});
        if(e.rotation)node.setRotationInParent(e.rotation,node.centerPointLocal);
        mapping.push({element_id:e.id,node_id:node.id});
        observed.push({element_id:e.id,node_id:node.id,opacity:node.opacity,bounds_local:node.boundsLocal?{x:node.boundsLocal.x,y:node.boundsLocal.y,width:node.boundsLocal.width,height:node.boundsLocal.height}:null,...(e.type==='text'?{text:node.fullContent.text}:{} )});
      }
    }catch(error){
      let cleanup='No document content created.';
      if(page){try{editor.documentRoot.pages.remove(page);cleanup='New page removed.';}catch{cleanup='New page cleanup failed; use Undo in Express to remove partial content.';}}
      throw new Error('Editable application failed: '+error.message+' '+cleanup);
    }
    result={applied:design.elements.length,application_id:++applicationSequence,page_id:page.id,mapping,observed,profile:capabilities(),fidelity:'host-unverified',warnings:['Live font units, glyph wrapping, rotation and image parity have not been verified. Prepared images preserve appearance, not original crop controls.']};
    });
    return result;
  }
  return {apply,capabilities,preflight};
}
