import profile from './capabilities.json' with {type:'json'};
export const capabilities = () => JSON.parse(JSON.stringify(profile));
const finite = value => typeof value === 'number' && Number.isFinite(value);

export function parseColor(value) {
  const text=String(value ?? '').trim().toLowerCase();
  let rgba;
  if(text==='transparent') rgba=[0,0,0,0];
  else if(/^#[\da-f]{3,4}$/.test(text)) rgba=[...text.slice(1)].map(c=>parseInt(c+c,16));
  else if(/^#[\da-f]{6}([\da-f]{2})?$/.test(text)) rgba=text.slice(1).match(/../g).map(c=>parseInt(c,16));
  else {
    const match=text.match(/^rgba?\(([^)]+)\)$/);
    if(!match) throw new Error('Use hex, rgb(), rgba(), or transparent colors.');
    const tokens=match[1].split(',').map(s=>s.trim());
    if(tokens.length<3||tokens.length>4||tokens.some(s=>!/^[-+]?(\d*\.)?\d+%?$/.test(s))) throw new Error('Invalid RGB color.');
    rgba=tokens.slice(0,3).map(s=>parseFloat(s)*(s.endsWith('%')?2.55:1));
    const alpha=tokens[3]??'1';rgba.push(parseFloat(alpha)*(alpha.endsWith('%')?2.55:255));
  }
  if(rgba.length===3)rgba.push(255);
  if(rgba.some(c=>!Number.isFinite(c)||c<0||c>255))throw new Error('Color channels outside supported range.');
  return {red:rgba[0]/255,green:rgba[1]/255,blue:rgba[2]/255,alpha:rgba[3]/255};
}

export function fontKey(element) {
  const suffix=element.font_weight==='bold' ? (element.font_style==='italic'?'BoldItalic':'Bold') : (element.font_style==='italic'?'Italic':'Regular');
  return String(element.font_family).replaceAll(' ','')+'-'+suffix;
}

export function preflight(design) {
  const errors=[];
  const issue=(id,property,message)=>errors.push({id,property,message});
  if(!design||!Array.isArray(design.elements)||!design.elements.length)return {supported:false,errors:[{id:null,property:'elements',message:'No design elements supplied.'}],profile:capabilities()};
  if(!design.canvas||!['width','height'].every(k=>Number.isInteger(design.canvas[k])&&design.canvas[k]>0&&design.canvas[k]<=16384)||design.canvas.width*design.canvas.height>32000000)issue(null,'canvas','Canvas must be positive integer pixels within render limits.');
  for(const key of Object.keys(design))if(!['canvas','elements'].includes(key))issue(null,key,'Keep metadata outside the canonical design.');
  const ids=new Set();
  for(const e of design.elements){
    if(!e||typeof e!=='object'){issue(null,'element','Expected a design element.');continue;}
    if(typeof e.id!=='string'||!e.id||ids.has(e.id))issue(e.id,'id','Every element needs a distinct nonempty ID.');
    ids.add(e.id);
    if(!profile.editable_types.includes(e.type)){issue(e.id,'type','Unsupported editable element.');continue;}
    const allowed=new Set([...profile.common_fields,...profile[e.type+'_fields']]);
    for(const key of Object.keys(e))if(!allowed.has(key))issue(e.id,key,'Unsupported property cannot be silently discarded.');
    for(const key of ['x','y','width','height'])if(!finite(e[key])||(key==='width'||key==='height')&&e[key]<=0)issue(e.id,key,'Geometry must be finite with positive dimensions.');
    if(e.rotation!==undefined&&!finite(e.rotation))issue(e.id,'rotation','Rotation must be finite.');
    if(e.opacity!==undefined&&(!finite(e.opacity)||e.opacity<0||e.opacity>1))issue(e.id,'opacity','Opacity must be in [0, 1].');
    if(e.z_index!==undefined&&!Number.isInteger(e.z_index))issue(e.id,'z_index','Stacking index must be an integer.');
    if(e.group_id!==undefined&&e.group_id!==null)issue(e.id,'group_id','Grouping is not yet verified; grouped designs cannot be applied faithfully.');
    if(e.type==='text'){
      if(typeof e.content!=='string'||typeof e.font_family!=='string'||!e.font_family.trim()||!finite(e.font_size)||e.font_size<1)issue(e.id,'text','Text content, font family and positive font size are required.');
      for(const [property,values] of [['font_weight',['regular','bold']],['font_style',['normal','italic']],['text_align',['left','center','right']]])if(e[property]!==undefined&&!values.includes(e[property]))issue(e.id,property,'Unsupported text style.');
      if(e.style_runs!==undefined&&(!Array.isArray(e.style_runs)||e.style_runs.length))issue(e.id,'style_runs','Rich-text ranges require a verified index/style adapter.');
      if(e.letter_spacing!==undefined&&e.letter_spacing!==0)issue(e.id,'letter_spacing','Custom letter spacing is unsupported.');
      if(e.line_height!==undefined&&e.line_height!==1.2)issue(e.id,'line_height','Custom line height is unsupported.');
      if(e.max_lines!==undefined&&e.max_lines!==null)issue(e.id,'max_lines','Line caps are unsupported.');
      if(e.capitalize!==undefined&&e.capitalize!==false)issue(e.id,'capitalize','Automatic capitalization is unsupported.');
      try{parseColor(e.color);}catch(error){issue(e.id,'color',error.message);}
    }
    if(e.type==='image'){
      if(typeof e.asset_id!=='string'||!e.asset_id)issue(e.id,'asset_id','Original asset reference is required.');
      if(e.fit!==undefined&&!['contain','cover','fill'].includes(e.fit))issue(e.id,'fit','Unknown image fit.');
    }
    if(e.type==='shape'){
      if(!profile.shape_types.includes(e.shape_type))issue(e.id,'shape_type','Only rectangles and ellipses are supported.');
      if((e.stroke_width??0)!==0||e.stroke_color!==undefined&&e.stroke_color!==null)issue(e.id,'stroke','Shape strokes are unsupported.');
      if((e.corner_radius??0)!==0)issue(e.id,'corner_radius','Rounded corners are unsupported.');
      try{parseColor(e.fill_color);}catch(error){issue(e.id,'fill_color',error.message);}
    }
  }
  return {supported:errors.length===0,errors,profile:capabilities()};
}

// Schema ranges are code-point based; a future host range adapter must translate
// them rather than applying Python offsets directly to UTF-16 JavaScript text.
export function codePointRangeToUtf16(text,start,end){
  const points=Array.from(text);
  if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<start||end>points.length)throw new Error('Invalid code-point range.');
  const offset=points.slice(0,start).join('').length;
  return {start:offset,length:points.slice(start,end).join('').length};
}
