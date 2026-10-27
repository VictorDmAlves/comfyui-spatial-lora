/* Resolve a *connected* image widget without executing arbitrary upstream nodes.
 * A native Load Image can be previewed instantly via ComfyUI's /view route.
 * Other IMAGE sources can be previewed from a bounded backend cache AFTER a run.
 */
function linkById(graph, id) {
    if (id == null) return null;
    return graph?.getLink?.(id) ?? graph?.links?.[id] ?? graph?.links?.get?.(id) ?? null;
}
function inputLink(graph, node, slot) {
    return node?.getInputLink?.(slot) ?? linkById(graph, node?.inputs?.[slot]?.link);
}
function sourceForImage(graph, node) {
    let slot = (node?.inputs || []).findIndex(i => i.name === "reference_image");
    if (slot < 0) return null;
    let link = inputLink(graph, node, slot);
    const seen = new Set();
    for (let depth = 0; link && depth < 12; depth++) {
        const origin = graph?.getNodeById?.(link.origin_id);
        if (!origin || seen.has(origin.id)) return { linked: true, kind: "computed" };
        seen.add(origin.id);
        const type = String(origin.comfyClass || origin.type || "");
        if (!/reroute/i.test(type)) return { linked: true, origin, type, kind: "source" };
        link = inputLink(graph, origin, 0);
    }
    return link ? { linked: true, kind: "computed" } : null;
}
function getImageName(raw) {
    if (typeof raw === "string") return raw;
    if (raw && typeof raw === "object") return raw.name ?? raw.filename ?? "";
    return "";
}
export function connectedReference(graph, node) {
    const source = sourceForImage(graph, node);
    if (!source) return { linked: false, kind: "none" };
    if (source.kind === "source") {
        const type = source.type;
        const isLoadImage = /^(LoadImage|LoadImageOutput|LoadImageMask)$/.test(type);
        if (isLoadImage) {
            const widget = source.origin.widgets?.find(w => w.name === "image");
            const raw = widget?.value;
            const name = getImageName(raw);
            if (name) {
                const folder = type === "LoadImageOutput" ? "output" : "input";
                const params = new URLSearchParams({ filename: name, type: folder });
                if (raw && typeof raw === "object") {
                    if (raw.subfolder) params.set("subfolder", String(raw.subfolder));
                    if (["input", "output", "temp"].includes(raw.type)) params.set("type", raw.type);
                }
                return { linked: true, kind: "load-image", path: `/view?${params}`, label: type };
            }
        }
    }
    return { linked: true, kind: "computed", label: source.type || "connected IMAGE" };
}

export function cachedReferencePath(node) {
    const id = String(node?.id ?? "");
    if (!/^[A-Za-z0-9_-]{1,128}$/.test(id)) return null;
    return `/spatial_lora/reference/${encodeURIComponent(id)}`;
}
