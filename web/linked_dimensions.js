/* Read graph-visible scalar inputs without executing arbitrary ComfyUI nodes.
 * Do not invent values for computed outputs. The Python node always receives
 * the real linked width/height at execution, regardless of this UI helper.
 */
const positiveInt = v => {
    if (typeof v === "boolean" || v == null || v === "") return null;
    const number = Number(v);
    return Number.isInteger(number) && number > 0 && number <= 8192 ? number : null;
};
const widgetValue = (node, name) => node?.widgets?.find(w => w.name?.toLowerCase() === name)?.value;
const widgetValues = node => (node?.widgets || []).filter(w => w.name && typeof w.value !== "undefined");
function linkById(graph, id) {
    if (id == null) return null;
    return graph?.getLink?.(id) || graph?.links?.[id] || graph?.links?.get?.(id) || null;
}
function inputLink(graph, node, slot) {
    const found = node?.getInputLink?.(slot);
    return found || linkById(graph, node?.inputs?.[slot]?.link);
}
function pickerDimensions(source) {
    const raw = String(widgetValue(source, "resolution") || "");
    const m = raw.match(/(\d{2,5})\s*[x×]\s*(\d{2,5})/i);
    if (!m) return null;
    const w = positiveInt(m[1]), h = positiveInt(m[2]);
    if (!w || !h) return null;
    const ow = positiveInt(widgetValue(source, "width_override"));
    const oh = positiveInt(widgetValue(source, "height_override"));
    return { width: ow || w, height: oh || h };
}
function valueFromOrigin(graph, origin, originSlot, depth, visited) {
    if (!origin || depth > 10 || visited.has(origin.id)) return null;
    visited.add(origin.id);
    const output = origin.outputs?.[originSlot] || {};
    const name = String(output.name || "").toLowerCase();
    const type = String(origin.comfyClass || origin.type || origin.title || "");
    // A pass-through/reroute preserves the input value, but has no value itself.
    if (/reroute/i.test(type)) {
        const link = inputLink(graph, origin, 0);
        if (!link) return null;
        return valueFromOrigin(graph, graph.getNodeById?.(link.origin_id), link.origin_slot, depth + 1, visited);
    }
    const isPicker = /empty\s*latent.*size\s*picker|SDXLEmptyLatentSizePicker|EmptyLatentSizePicker/i.test(type)
        || (widgetValue(origin, "resolution") !== undefined && widgetValue(origin, "width_override") !== undefined);
    if (isPicker) {
        const dims = pickerDimensions(origin);
        if (dims && (name === "width" || name === "height")) return dims[name];
        // In the usual Empty Latent Size Picker, outputs are LATENT, width, height.
        if (dims && originSlot === 1) return dims.width;
        if (dims && originSlot === 2) return dims.height;
    }
    // Regular numeric constant nodes, including converted ComfyUI primitives.
    if (String(output.type).toUpperCase() === "INT" || /primitive|integer|\bint\b|constant/i.test(type)) {
        for (const key of [name, "value", "int", "integer", "number"]) {
            if (!key) continue;
            const found = positiveInt(widgetValue(origin, key));
            if (found !== null) return found;
        }
        const numeric = widgetValues(origin)
            .filter(w => ["value", "int", "integer", "number"].includes(w.name.toLowerCase()));
        if (numeric.length === 1) return positiveInt(numeric[0].value);
    }
    // Pass-through controls exposing a widget with the output's exact name.
    if (["width", "height"].includes(name)) {
        const value = positiveInt(widgetValue(origin, name));
        if (value !== null && !widgetValue(origin, "resolution")) return value;
    }
    return null;
}

export function resolveEditorDimensions(graph, node) {
    const widgets = Object.fromEntries((node.widgets || []).map(w => [w.name, w]));
    const resolved = {}, unresolved = [];
    let linksFound = 0;
    for (const key of ["width", "height"]) {
        const slot = (node.inputs || []).findIndex(i => i.name === key);
        const link = slot >= 0 ? inputLink(graph, node, slot) : null;
        if (link) {
            linksFound++;
            const origin = graph?.getNodeById?.(link.origin_id);
            const v = valueFromOrigin(graph, origin, link.origin_slot, 0, new Set());
            if (v !== null) resolved[key] = v;
            else {
                unresolved.push(key);
                resolved[key] = positiveInt(widgets[key]?.value) || 1024;
            }
        } else resolved[key] = positiveInt(widgets[key]?.value) || 1024;
    }
    return { width: resolved.width, height: resolved.height,
        resolvedFromLinks: linksFound > 0 && !unresolved.length,
        unresolved, linked: linksFound > 0 };
}
