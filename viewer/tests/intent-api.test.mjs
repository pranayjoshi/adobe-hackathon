import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const ts=require('typescript');
function load(file){
  const compiled=ts.transpileModule(readFileSync(new URL(file,import.meta.url),'utf8'),{
    compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2022},
  }).outputText;
  const module={exports:{}};new Function('module','exports','require',compiled)(module,module.exports,require);return module.exports;
}
const source={canvas:{width:100,height:100},elements:[
  {id:'background',type:'shape',x:0,y:0,width:100,height:100},
  {id:'copy',type:'text',content:'Keep these exact words',x:10,y:10,width:80,height:30},
]};

test('intent travels separately from source and old generate callers remain valid',async()=>{
  const oldFetch=globalThis.fetch;
  const requests=[];
  globalThis.fetch=async(path,options)=>{
    requests.push({path,body:options.body?JSON.parse(options.body):undefined});
    return {ok:true,json:async()=>path==='/api/jobs'?{id:'run'}:{state:'done',progress:1,events:[],result:{results:[]}}};
  };
  try{
    const {api}=load('../src/api.ts');const before=structuredClone(source);
    const intent={roles:{copy:'headline'},group_modes:{stack:'rigid'}};
    await api.generate(source,[],'balanced',{},false,{},undefined,undefined,intent);
    assert.deepEqual(requests[0].body.analysis_overrides,intent);
    assert.deepEqual(requests[0].body.source,before);
    assert.equal('analysis_overrides' in requests[0].body.source,false);
    assert.deepEqual(source,before);
    await api.generate(source,[],'balanced');
    assert.deepEqual(requests[2].body.analysis_overrides,{roles:{},group_modes:{}});
  }finally{globalThis.fetch=oldFetch;}
});

test('inspector exposes only actual content IDs and engine-declared group transitions',()=>{
  const React=require('react');const {renderToStaticMarkup}=require('react-dom/server');
  const {IntentInspector}=load('../src/components/IntentInspector.tsx');
  const analysis={roles:{background:'background',copy:'body'},blocks:[{members:['copy']}],stickers:[],accents:[],
    graph:{nodes:{copy:{role_source:'heuristic',role_confidence:.8}},group_policies:[
      {group_id:'stack',member_ids:['copy'],mode:'reflowable',authority:'inferred',kind:'display_stack',editable_modes:['reflowable','rigid']},
    ]}};
  const props={source,analysis,value:{roles:{},group_modes:{}},onChange:()=>{},semantic:true,pending:true,busy:false};
  const html=renderToStaticMarkup(React.createElement(IntentInspector,props));
  assert.match(html,/Purpose of copy/);assert.doesNotMatch(html,/Purpose of background/);
  assert.match(html,/Behavior of stack/);assert.doesNotMatch(html,/value="support"/);
  assert.match(html,/pending for the next generation/);
  const readonly=renderToStaticMarkup(React.createElement(IntentInspector,{...props,semantic:false}));
  assert.doesNotMatch(readonly,/aria-label="Behavior of stack"/);
  assert.match(readonly,/Enable Semantic layout/);
});
