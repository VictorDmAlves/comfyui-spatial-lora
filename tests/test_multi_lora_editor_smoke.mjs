import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mkdtemp, mkdir, copyFile, writeFile, rm } from 'node:fs/promises';

const root = path.dirname(fileURLToPath(import.meta.url));
class FakeElement {
    constructor(tag) {
        this.tag = tag; this.children = []; this.style = { cssText: '' };
        this.events = new Map(); this.textContent = ''; this.value = ''; this.checked = false;
        this.width = 540; this.height = 540; this.files = null;
    }
    append(...items) { this.children.push(...items); }
    appendChild(item) { this.children.push(item); return item; }
    replaceChildren(...items) { this.children = items; }
    setAttribute(key, value) { this[key] = value; }
    addEventListener(type, fn) { this.events.set(type, fn); }
    removeEventListener(type) { this.events.delete(type); }
    dispatch(type, value = {}) { this.events.get(type)?.({ target: this, ...value }); }
    remove() { this.removed = true; }
    focus() { globalThis.document.activeElement = this; }
    matches(sel) { return this.tag === 'input' && sel.includes('input'); }
    getBoundingClientRect() { return { left: 0, top: 0, width: this.width, height: this.height }; }
    setPointerCapture() {}
    getContext() {
        const ctx = { measureText(text) { return { width: String(text).length * 7 }; },
            createImageData(width, height) { return { data: new Uint8ClampedArray(width * height * 4) }; } };
        for (const k of ['clearRect','fillRect','drawImage','strokeRect','beginPath','moveTo','lineTo','stroke',
            'rect','clip','save','restore','setLineDash','fillText','putImageData']) ctx[k] = () => {};
        return ctx;
    }
}
function all(element, predicate, out = []) {
    if (element && typeof element === 'object') {
        if (predicate(element)) out.push(element);
        for (const child of element.children || []) all(child, predicate, out);
    }
    return out;
}
const find = (parent, tag, label) => all(parent, el => el.tag === tag && el.textContent === label)[0];
function makeNode() {
    const layout = JSON.stringify({canvas:{width:1024,height:1024},regions:[
        {id:1,name:'Gigi',enabled:true,x:0,y:100,w:512,h:924,feather:8},
        {id:2,name:'Michelle',enabled:true,x:512,y:100,w:512,h:924,feather:8},
        {id:3,name:'Region 3',enabled:false,x:90,y:90,w:200,h:200,feather:8},
    ]});
    const loras = ['None','gigi.safetensors','michelle.safetensors','new_character.safetensors'];
    const widgets = [
        {name:'width',value:1024},{name:'height',value:1024},{name:'regions_json',type:'text',value:layout},
        ...[1,2,3].flatMap((id)=>[
            {name:`lora_${id}`,value:id === 1?'gigi.safetensors':id === 2?'michelle.safetensors':'None',options:{values:loras}},
            {name:`strength_${id}`,value:1.0},
        ]),
    ];
    return {id:42,type:'SpatialMultiLoRA',comfyClass:'SpatialMultiLoRA',widgets,inputs:[],size:[400,230],
        addWidget(type,name,value,callback) {this.widgets.push({type,name,value,callback});},
        setDirtyCanvas(){this.dirty=true;},setSize(size){this.size=size;},computeSize(){return [400,300];}};
}

async function makeEnvironment() {
    const tmp = await mkdtemp(path.join(os.tmpdir(),'multi-lora-v06-'));
    const ext = path.join(tmp,'extensions','spatial');
    const scripts = path.join(tmp,'scripts');
    await mkdir(ext,{recursive:true}); await mkdir(scripts,{recursive:true});
    for (const name of ['multi_lora_editor.js','mask_preview.js','bbox_geometry.js','linked_dimensions.js','reference_image.js'])
        await copyFile(path.join(root,'..','web',name),path.join(ext,name));
    await writeFile(path.join(tmp,'package.json'),' {"type":"module"} ');
    await writeFile(path.join(scripts,'app.js'),'export const app = globalThis.__testApp;');
    await writeFile(path.join(scripts,'api.js'),'export const api = { apiURL: (path) => `/api${path}` };');
    let extension;
    globalThis.__testApp={graph:{setDirtyCanvas(){}},registerExtension(e){extension=e;}};
    globalThis.document={body:new FakeElement('body'),activeElement:null,
        createElement:tag=>new FakeElement(tag),createTextNode:text=>({textContent:text}),
        addEventListener(){},removeEventListener(){}};
    await import(pathToFileURL(path.join(ext,'multi_lora_editor.js')).href);
    return {tmp,extension,cleanup:async()=>{
        delete globalThis.__testApp;delete globalThis.document;
        await rm(tmp,{recursive:true,force:true});
    }};
}

test('v0.6 editor binds each LoRA to a numbered region; Apply commits, Cancel discards',async()=>{
    const env=await makeEnvironment();
    try {
        const n=makeNode();
        await env.extension.nodeCreated(n);
        assert.equal(n.widgets.find(w=>w.name==='regions_json').type,'hidden');
        assert.equal(n.widgets.find(w=>w.name==='Open spatial editor'), undefined);
        const launcher=n.widgets.find(w=>w.name==='Open Spatial Editor');
        assert.ok(launcher);
        launcher.callback();
        const root=document.body;
        const selects=all(root,e=>e.tag==='select');
        assert.equal(selects.length,2,'mode select plus LoRA select');
        const loraSelect=selects[1];
        assert.equal(loraSelect.value,'gigi.safetensors');
        loraSelect.value='new_character.safetensors'; loraSelect.dispatch('change');
        const strength=all(root,e=>e.tag==='input'&&e.step==='0.05')[0];
        strength.value='1.15';strength.dispatch('change');
        assert.equal(n.widgets.find(w=>w.name==='lora_1').value,'gigi.safetensors','stage values until Apply');
        find(root,'button','Apply regions').dispatch('click');
        assert.equal(n.widgets.find(w=>w.name==='lora_1').value,'new_character.safetensors');
        assert.equal(n.widgets.find(w=>w.name==='strength_1').value,1.15);
        assert.equal(JSON.parse(n.widgets.find(w=>w.name==='regions_json').value).canvas.width,1024);
        assert.equal(n.dirty,true);
        launcher.callback();
        const selectAgain=all(document.body,e=>e.tag==='select')[1];
        selectAgain.value='gigi.safetensors'; selectAgain.dispatch('change');
        find(document.body,'button','Cancel').dispatch('click');
        assert.equal(n.widgets.find(w=>w.name==='lora_1').value,'new_character.safetensors');
    } finally {await env.cleanup();}
});

test('v0.7 preserves a LoRA selected in the node before enabling its region, then Remove clears the node',async()=>{
    const env=await makeEnvironment();
    try {
        const n=makeNode();
        n.widgets.find(w=>w.name==='lora_3').value='new_character.safetensors';
        await env.extension.nodeCreated(n);
        assert.equal(n.widgets.find(w=>w.name==='Open Spatial Editor')?.name,'Open Spatial Editor');
        n.widgets.find(w=>w.name==='Open Spatial Editor').callback();
        const root=document.body;
        const regionButtons=all(root,e=>e.tag==='button' && /^● 3\./.test(e.textContent));
        assert.equal(regionButtons.length,1,'setting LoRA in node should enable region 3 in editor staging');
        assert.match(regionButtons[0].textContent,/new_character\.safetensors/);
        assert.match(regionButtons[0].style.cssText,/text-overflow:ellipsis/);
        assert.equal(find(root,'button','Duplicate'), undefined);
        assert.equal(find(root,'button','Import existing BBoxes'), undefined);
        regionButtons[0].dispatch('click');
        const selects=all(root,e=>e.tag==='select');
        assert.equal(selects[1].value,'new_character.safetensors');
        find(root,'button','Remove').dispatch('click');
        find(root,'button','Apply regions').dispatch('click');
        assert.equal(n.widgets.find(w=>w.name==='lora_3').value,'None');
        assert.equal(n.widgets.find(w=>w.name==='strength_3').value,1.0);
        assert.equal(JSON.parse(n.widgets.find(w=>w.name==='regions_json').value).regions[2].enabled,false);
    } finally {await env.cleanup();}
});

test('v0.7 editor reads current graph-connected resolution instead of original local input defaults',async()=>{
    const env=await makeEnvironment();
    try {
        const n=makeNode();
        const picker={id:77,type:'SDXLEmptyLatentSizePicker',
            widgets:[{name:'resolution',value:'1216x832 (1.46)'},{name:'width_override',value:0},{name:'height_override',value:0}],
            outputs:[{name:'LATENT'},{name:'width',type:'INT'},{name:'height',type:'INT'}]};
        n.inputs=[{name:'width',link:11},{name:'height',link:12}];
        env.extension.nodeCreated(n);
        globalThis.__testApp.graph={
            links:{11:{origin_id:77,origin_slot:1},12:{origin_id:77,origin_slot:2}},
            getNodeById(id){return id===77?picker:null;},setDirtyCanvas(){},
        };
        n.widgets.find(w=>w.name==='Open Spatial Editor').callback();
        const dialog=all(document.body,e=>e.tag==='div'&&String(e.textContent).includes('Multi-LoRA Editor v0.7'))[0];
        assert.match(dialog.textContent,/1216 × 832/);
        find(document.body,'button','Apply regions').dispatch('click');
        const saved=JSON.parse(n.widgets.find(w=>w.name==='regions_json').value);
        assert.deepEqual(saved.canvas,{width:1216,height:832});
        picker.widgets[0].value='896x1152 (0.78)';
        n.widgets.find(w=>w.name==='Open Spatial Editor').callback();
        const second=all(document.body,e=>e.tag==='div'&&String(e.textContent).includes('Multi-LoRA Editor v0.7')).at(-1);
        assert.match(second.textContent,/896 × 1152/);
        find(document.body,'button','Cancel').dispatch('click');
        assert.deepEqual(JSON.parse(n.widgets.find(w=>w.name==='regions_json').value).canvas,{width:1216,height:832});
    } finally {await env.cleanup();}
});


test('v0.7.1 editor loads connected Load Image instantly; manual reference remains available',async()=>{
    const env=await makeEnvironment();
    const oldImage=globalThis.Image;
    const loaded=[];
    globalThis.Image=class {
        set src(url) { loaded.push(url); this.onload?.(); }
    };
    try {
        const n=makeNode();
        const load={id:17,type:'LoadImage',widgets:[{name:'image',value:'photo test.png'}],
                    outputs:[{name:'IMAGE',type:'IMAGE'}]};
        n.inputs=[{name:'reference_image',link:222}];
        globalThis.__testApp.graph={
            links:{222:{origin_id:17,origin_slot:0}},
            getNodeById(id){return id===17?load:null;},setDirtyCanvas(){}
        };
        env.extension.nodeCreated(n);
        n.widgets.find(w=>w.name==='Open Spatial Editor').callback();
        assert.equal(loaded.length,1,'direct Load Image preview needs no sampling');
        assert.match(loaded[0],/\/api\/view\?filename=photo\+test\.png&type=input/);
        assert.ok(find(document.body,'button','Use connected image'));
        const file=all(document.body,e=>e.tag==='input'&&e.type==='file')[0];
        assert.ok(file,'manual image loader stays available');
        const status=all(document.body,e=>e.tag==='span' && String(e.textContent).includes('automatic preview'))[0];
        assert.ok(status,'connected image reported as loaded');
        find(document.body,'button','Clear image').dispatch('click');
        find(document.body,'button','Use connected image').dispatch('click');
        assert.equal(loaded.length,2);
        find(document.body,'button','Cancel').dispatch('click');
    } finally {globalThis.Image=oldImage;await env.cleanup();}
});

test('v0.7.1 computed IMAGE input uses cache from most recent execution',async()=>{
    const env=await makeEnvironment();
    const oldImage=globalThis.Image;
    const loaded=[];
    globalThis.Image=class {set src(url){loaded.push(url);this.onload?.();}};
    try {
        const n=makeNode();
        const source={id:21,type:'ImageScale',outputs:[{name:'IMAGE',type:'IMAGE'}]};
        n.inputs=[{name:'reference_image',link:223}];
        globalThis.__testApp.graph={links:{223:{origin_id:21,origin_slot:0}},
            getNodeById(id){return id===21?source:null;},setDirtyCanvas(){}};
        env.extension.nodeCreated(n);
        n.widgets.find(w=>w.name==='Open Spatial Editor').callback();
        assert.equal(loaded.length,1);
        assert.match(loaded[0],/\/api\/spatial_lora\/reference\/42\?v=/);
        find(document.body,'button','Cancel').dispatch('click');
    } finally {globalThis.Image=oldImage;await env.cleanup();}
});

test('v0.7.1 native Load Image through reroute is resolved before sampling', async()=>{
    const env=await makeEnvironment();
    const original=globalThis.Image;
    const calls=[];
    globalThis.Image=class {set src(url){calls.push(url);this.onload?.();}};
    try {
        const n=makeNode();
        const load={id:16,type:'LoadImage',widgets:[{name:'image',value:'portrait.png'}],outputs:[{name:'IMAGE',type:'IMAGE'}]};
        const reroute={id:17,type:'Reroute',inputs:[{name:'',link:42}]};
        n.inputs=[{name:'reference_image',link:43}];
        globalThis.__testApp.graph={links:{42:{origin_id:16,origin_slot:0},43:{origin_id:17,origin_slot:0}},
            getNodeById(id){return id===16?load:id===17?reroute:null;},setDirtyCanvas(){}};
        env.extension.nodeCreated(n);
        n.widgets.find(w=>w.name==='Open Spatial Editor').callback();
        assert.equal(calls[0],'/api/view?filename=portrait.png&type=input');
        find(document.body,'button','Cancel').dispatch('click');
    } finally {globalThis.Image=original;await env.cleanup();}
});
