import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveEditorDimensions } from '../web/linked_dimensions.js';

function connected(originW, originH, slotW = 0, slotH = 0) {
    const target = {widgets:[{name:'width',value:1024},{name:'height',value:1024}],
        inputs:[{name:'width',link:101},{name:'height',link:102}]};
    const graph = {links:{
        101:{origin_id:originW.id,origin_slot:slotW},
        102:{origin_id:originH.id,origin_slot:slotH},
    },getNodeById(id){return [originW,originH].find(n=>n.id===id);}};
    return [graph,target];
}
test('reads live width / height from linked Empty Latent Size Picker including overrides',()=>{
    const picker={id:4,type:'SDXLEmptyLatentSizePicker',
        widgets:[{name:'resolution',value:'1216x832 (1.46)'},{name:'width_override',value:0},{name:'height_override',value:0}],
        outputs:[{name:'LATENT'},{name:'width',type:'INT'},{name:'height',type:'INT'}]};
    const [graph,target] = connected(picker,picker,1,2);
    assert.deepEqual(resolveEditorDimensions(graph,target),
        {width:1216,height:832,resolvedFromLinks:true,unresolved:[],linked:true});
    picker.widgets[1].value=640;
    assert.equal(resolveEditorDimensions(graph,target).width,640);
    picker.widgets[0].value='880x1152 (0.76)';
    assert.equal(resolveEditorDimensions(graph,target).height,1152);
});
test('reads numeric constants and does not claim arbitrary computed INT as known',()=>{
    const x={id:11,type:'PrimitiveNode',widgets:[{name:'value',value:896}],outputs:[{name:'INT',type:'INT'}]};
    const y={id:12,type:'IntConstant',widgets:[{name:'value',value:1344}],outputs:[{name:'INT',type:'INT'}]};
    const [graph,target] = connected(x,y);
    assert.equal(resolveEditorDimensions(graph,target).width,896);
    assert.equal(resolveEditorDimensions(graph,target).height,1344);
    y.type='UnknownMathComputation';y.widgets=[];
    assert.deepEqual(resolveEditorDimensions(graph,target).unresolved,['height']);
});
test('supports simple reroute and widget defaults for unlinked dimensions',()=>{
    const origin={id:1,type:'PrimitiveNode',widgets:[{name:'value',value:768}],outputs:[{name:'INT',type:'INT'}]};
    const reroute={id:2,type:'Reroute',inputs:[{name:'',link:55}],outputs:[{name:'',type:'INT'}]};
    const target={widgets:[{name:'width',value:1024},{name:'height',value:900}],inputs:[{name:'width',link:99},{name:'height',link:null}]};
    const graph={links:{55:{origin_id:1,origin_slot:0},99:{origin_id:2,origin_slot:0}},getNodeById(id){return [origin,reroute].find(n=>n.id===id)}};
    assert.deepEqual(resolveEditorDimensions(graph,target),{width:768,height:900,resolvedFromLinks:true,unresolved:[],linked:true});
});
