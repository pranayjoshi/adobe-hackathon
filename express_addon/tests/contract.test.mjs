import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createAdapter} from '../src/adapter.js';
import {capabilities,codePointRangeToUtf16,fontKey,parseColor,preflight} from '../src/capabilities.js';

function fixture(){return {canvas:{width:300,height:200},elements:[
  {id:'background',type:'shape',shape_type:'rectangle',x:0,y:0,width:300,height:200,fill_color:'#000',opacity:.5,z_index:0},
  {id:'headline',type:'text',content:'A😀B',font_family:'Solstice Sans',font_size:24,font_weight:'bold',font_style:'italic',color:'rgba(255, 255, 255, 1)',text_align:'center',x:20,y:20,width:180,height:50,z_index:0}
]};}
function mockHost({missingFont=false,failAppend=false,failCleanup=false}={}){
  const state={pages:0,removed:0,nodes:[],styles:[]};let next=0;
  const node=()=>({id:'node-'+ ++next,fullContent:{text:'',applyCharacterStyles(style,range){state.styles.push({style,range});}},boundsLocal:{x:0,y:0,width:10,height:10},topLeftLocal:{x:0,y:0},centerPointLocal:{x:5,y:5},setPositionInParent(point){this.position=point;},setRotationInParent(rotation){this.rotation=rotation;}});
  const editor={documentRoot:{pages:{addPage(){state.pages++;return {id:'page-1',artboards:{first:{children:{append(n){if(failAppend)throw new Error('append failed');state.nodes.push(n);}}}}};},remove(){if(failCleanup)throw new Error('cannot remove');state.removed++;}}},createRectangle:node,createEllipse:node,createText(text){const n=node();n.fullContent.text=text;return n;},makeColorFill(color){return color;},async loadBitmapImage(){return {bitmap:true};},createImageContainer:node};
  editor.context={insertionParent:{}};editor.keepContentActiveDuringAsync=async(target,prepare,mutate)=>{await prepare();mutate();};
  const constants={TextLayout:{autoHeight:'autoHeight'},TextAlignment:{left:'left',right:'right',center:'center'}};
  const fonts={async fromPostscriptName(name){return missingFont?{availableForEditing:false}:{availableForEditing:true,name};}};
  return {state,adapter:createAdapter({editor,constants,fonts})};
}

test('capability profile blocks unsupported groups, spans, strokes and unknown loss',()=>{
  const design=fixture();assert.equal(preflight(design).supported,true);
  design.elements[1].group_id='outer/inner';design.elements[1].style_runs=[{start:1,end:2,color:'#fff'}];
  design.elements[0].stroke_width=2;design.elements[0].stroke_color='#fff';design.elements[1].shadow={x:3};
  const report=preflight(design);assert.equal(report.supported,false);
  for(const property of ['group_id','style_runs','stroke','shadow'])assert.ok(report.errors.some(e=>e.property===property),property);
  assert.equal(capabilities().host_verified,false);
  assert.deepEqual(codePointRangeToUtf16('A😀B',1,2),{start:1,length:2});
  assert.equal(fontKey(fixture().elements[1]),'SolsticeSans-BoldItalic');
  assert.deepEqual(parseColor('#f008'),{red:1,green:0,blue:0,alpha:136/255});
  assert.throws(()=>parseColor('rgba(256,0,0,1)'));
});

test('failed resources or unsupported properties produce no page mutation',async()=>{
  const host=mockHost({missingFont:true});await assert.rejects(host.adapter.apply({design:fixture()}),/Editable font unavailable/);assert.equal(host.state.pages,0);
  const design=fixture();design.elements.push({id:'img',type:'image',asset_id:'original.png',x:0,y:0,width:10,height:10});
  const imageHost=mockHost();await assert.rejects(imageHost.adapter.apply({design}),/image unavailable/);assert.equal(imageHost.state.pages,0);
  design.elements[0].corner_radius=4;await assert.rejects(imageHost.adapter.apply({design}),/Rounded corners/);assert.equal(imageHost.state.pages,0);
});

test('supported apply retains identity, content, paint order and reports actual observations',async()=>{
  const {adapter,state}=mockHost();const design=fixture(),before=structuredClone(design);
  const result=await adapter.apply({design});assert.deepEqual(design,before);
  assert.equal(result.applied,2);assert.deepEqual(result.mapping.map(e=>e.element_id),['background','headline']);
  assert.equal(state.nodes[0].opacity,.5);assert.equal(state.nodes[1].textAlignment,'center');
  assert.equal(state.styles[0].style.fontSize,18);assert.equal(state.styles[0].style.font.name,'SolsticeSans-BoldItalic');
  assert.equal(result.observed[1].text,'A😀B');assert.equal(result.fidelity,'host-unverified');
});

test('post-mutation failure removes its page, or supplies explicit Undo recovery',async()=>{
  let host=mockHost({failAppend:true});await assert.rejects(host.adapter.apply({design:fixture()}),/New page removed/);assert.equal(host.state.pages,1);assert.equal(host.state.removed,1);
  host=mockHost({failAppend:true,failCleanup:true});await assert.rejects(host.adapter.apply({design:fixture()}),/use Undo/);
});
