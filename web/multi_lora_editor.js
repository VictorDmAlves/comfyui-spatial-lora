import { app } from "../../scripts/app.js";
import { clamp, normalizeRect, scaleRect, intersection, collectGuides, snapMove, snapResize } from "./bbox_geometry.js";
import { SLOT_COLORS, renderMaskPixels } from "./mask_preview.js";
import { resolveEditorDimensions } from "./linked_dimensions.js";
import { connectedReference, cachedReferencePath } from "./reference_image.js";
import { api } from "../../scripts/api.js";

const MAX = 3;
const MAX_FEATHER = 128;
const DEFAULTS = [
    { id: 1, name: "Region 1", enabled: true, x: 90, y: 90, w: 380, h: 720, feather: 8 },
    { id: 2, name: "Region 2", enabled: false, x: 550, y: 90, w: 380, h: 720, feather: 8 },
    { id: 3, name: "Region 3", enabled: false, x: 350, y: 90, w: 300, h: 720, feather: 8 },
];
const css = {
    button: "border:1px solid #607082;border-radius:6px;padding:7px 10px;color:#f6f8fc;background:#384453;cursor:pointer",
    input: "width:100%;box-sizing:border-box;padding:7px;background:#171d26;border:1px solid #647183;border-radius:6px;color:#f6f8fc;font:12px monospace",
};
const isMulti = node => node?.comfyClass === "SpatialMultiLoRA" || node?.type === "SpatialMultiLoRA";
function el(tag, style = "", text) {
    const result = document.createElement(tag);
    if (style) result.style.cssText = style;
    if (text !== undefined) result.textContent = text;
    return result;
}
function textInput(type, title) {
    const input = el("input", css.input);
    input.type = type;
    if (title) input.title = title;
    return input;
}
function regionColor(id) {
    return `rgb(${SLOT_COLORS[id - 1].join(",")})`;
}
function normalizeRegion(r, width, height, id) {
    const b = normalizeRect(r, width, height);
    return {
        id, enabled: r.enabled === true, name: String(r.name || `Region ${id}`).slice(0, 64),
        ...b, feather: clamp(Math.round(Number(r.feather) || 0), 0, MAX_FEATHER),
    };
}
function loadRegions(raw, width, height) {
    const parsed = JSON.parse(raw);
    const data = Array.isArray(parsed) ? parsed : parsed?.regions;
    if (!Array.isArray(data) || data.length > MAX) throw new Error("Invalid regions array");
    const sourceW = Array.isArray(parsed) ? width : Number(parsed.canvas?.width);
    const sourceH = Array.isArray(parsed) ? height : Number(parsed.canvas?.height);
    if (!Number.isFinite(sourceW) || !Number.isFinite(sourceH) || sourceW < 64 || sourceH < 64)
        throw new Error("Invalid saved layout canvas size");
    const regions = DEFAULTS.map((r, i) => normalizeRegion({ ...r, enabled: false }, width, height, i + 1));
    const ids = new Set();
    for (const candidate of data) {
        const id = Number(candidate?.id);
        if (![1, 2, 3].includes(id) || ids.has(id)) throw new Error("Region id must be unique and within 1..3");
        ids.add(id);
        const oldRect = normalizeRegion(candidate, sourceW, sourceH, id);
        const scaled = (sourceW !== width || sourceH !== height)
            ? normalizeRect({
                x: Math.round(oldRect.x * width / sourceW),
                y: Math.round(oldRect.y * height / sourceH),
                w: Math.max(1, Math.round((oldRect.x + oldRect.w) * width / sourceW)
                    - Math.round(oldRect.x * width / sourceW)),
                h: Math.max(1, Math.round((oldRect.y + oldRect.h) * height / sourceH)
                    - Math.round(oldRect.y * height / sourceH)),
            }, width, height) : oldRect;
        regions[id - 1] = { ...oldRect, ...scaled,
            feather: clamp(Math.round(oldRect.feather * Math.sqrt(width * height / (sourceW * sourceH))), 0, MAX_FEATHER) };
    }
    return regions;
}
function saveRegions(regions, width, height) {
    return JSON.stringify({ canvas: { width, height }, regions: regions.map(r => ({
        id: r.id, name: r.name, enabled: r.enabled,
        x: r.x, y: r.y, w: r.w, h: r.h, feather: r.feather,
    })) });
}
function openEditor(node) {
    const widgets = Object.fromEntries((node.widgets || []).map(w => [w.name, w]));
    const config = widgets.regions_json;
    if (!config || !widgets.width || !widgets.height) {
        alert("Spatial LoRA v0.7: required widgets missing from Multi LoRA node."); return;
    }
    // Read graph-visible upstream values at editor OPEN, not the stale hidden
    // input widget defaults. Computed/unavailable sources are reported below.
    const linkedSize = resolveEditorDimensions(app.graph, node);
    const width = linkedSize.width;
    const height = linkedSize.height;
    let regions;
    try { regions = loadRegions(config.value, width, height); }
    catch (error) { alert(`Spatial LoRA: invalid regions_json. ${error.message}`); return; }
    // Stage widget changes locally so Cancel never mutates the workflow.
    const pending = {};
    for (let id = 1; id <= MAX; id++) {
        pending[id] = {
            lora: String(widgets[`lora_${id}`]?.value ?? "None"),
            strength: Number(widgets[`strength_${id}`]?.value ?? 1.0),
        };
    }
    // A LoRA chosen directly on the ComfyUI node should activate its region
    // in the editor, rather than silently showing it as empty/inactive.
    for (let id = 1; id <= MAX; id++) {
        if (pending[id].lora && pending[id].lora !== "None") regions[id - 1].enabled = true;
    }
    let selectedId = regions.find(r => r.enabled)?.id ?? null;
    let mode = "overlay", reference = null, objectUrl = null, closed = false;
    let referenceLoadId = 0;
    const connectedImage = connectedReference(app.graph, node);
    let drag = null, activeGuides = { guideX: null, guideY: null };
    let snapFeedback = "";
    let raf = null;
    const overlay = el("div", "position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;background:#000c;padding:8px;box-sizing:border-box");
    const panel = el("div", "color:#f1f5f9;background:#232831;border:1px solid #566374;border-radius:12px;padding:11px 13px;box-sizing:border-box;width:min(1060px,calc(100vw - 16px));max-height:calc(100dvh - 16px);overflow:auto;font:12px/1.3 sans-serif;box-shadow:0 16px 55px #000c");
    panel.setAttribute("role", "dialog"); panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-label", "Spatial LoRA unified multi-LoRA editor");
    panel.append(el("div", "font-size:17px;font-weight:700;margin-bottom:3px", `Spatial LoRA · Multi-LoRA Editor v0.7.1 (${width} × ${height})`));
    panel.append(el("div", "font-size:11px;color:#b8c7d7;margin-bottom:6px",
        "Select a region and drag it, resize with the handles, or draw on empty space. Snap is optional (hold Alt to bypass). Each region controls its own LoRA, strength and mask. MASK output slots are numbered 1–3."));
    const controls = el("div", "display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#303946;padding:6px;border-radius:6px;margin-bottom:7px");
    const modeLabel = el("label", "display:flex;align-items:center;gap:6px", "Preview");
    const modeSelect = el("select", css.input + ";width:auto");
    for (const [value, label] of [["overlay", "Overlay + Feather"], ["mask", "Mask (active region)"], ["heatmap", "Heatmap (all regions)"]]) {
        const opt = el("option", "", label); opt.value = value; modeSelect.append(opt);
    }
    modeSelect.value = mode;
    modeLabel.append(modeSelect);
    const snapLabel = el("label", "display:flex;align-items:center;gap:6px;cursor:pointer");
    const snapCheck = el("input"); snapCheck.type = "checkbox"; snapCheck.checked = true;
    snapLabel.append(snapCheck, document.createTextNode("Snap to edges / centers"));
    const overlapLabel = el("label", "display:flex;align-items:center;gap:6px;cursor:pointer");
    const overlapCheck = el("input"); overlapCheck.type = "checkbox"; overlapCheck.checked = true;
    overlapLabel.append(overlapCheck, document.createTextNode("Highlight overlap"));
    const rangeLabel = el("label", "display:flex;align-items:center;gap:6px", "Snap distance");
    rangeLabel.title = "Distance in displayed canvas pixels. Drag near another rectangle or the canvas edge to snap. Hold Alt to bypass.";
    const snapTolerance = el("input", "width:90px;cursor:pointer"); snapTolerance.type = "range";
    snapTolerance.value = "10"; snapTolerance.min = "1"; snapTolerance.max = "30"; snapTolerance.step = "1";
    const snapReadout = el("span", "min-width:35px;font:11px monospace", "10 px");
    rangeLabel.append(snapTolerance, snapReadout);
    controls.append(modeLabel, snapLabel, overlapLabel, rangeLabel); panel.append(controls);

    const cols = el("div", "display:flex;gap:11px;align-items:flex-start;flex-wrap:wrap");
    const left = el("div", "flex:1 1 520px;min-width:0");
    const canvas = el("canvas", "display:block;max-width:100%;height:auto;border:1px solid #6b7b90;border-radius:4px;cursor:crosshair;touch-action:none;background:#222a33");
    const viewportH = typeof window !== "undefined" && Number.isFinite(window.innerHeight) ? window.innerHeight : 900;
    const maxCanvasH = Math.max(250, Math.min(520, viewportH - 355));
    const scale = Math.min(665 / width, maxCanvasH / height, 1);
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    canvas.tabIndex = 0;
    const ctx = canvas.getContext("2d");
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = canvas.width; maskCanvas.height = canvas.height;
    const maskCtx = maskCanvas.getContext("2d");
    if (!ctx || !maskCtx) { alert("Spatial LoRA requires Canvas 2D support"); return; }
    const diagnostics = el("div", "font:11px/1.3 monospace;color:#cdd9e9;margin-top:4px");
    const warnings = el("div", "padding:6px;background:#334036;border-radius:6px;margin-top:5px;font-size:11px;color:#cfe8d8;min-height:18px");
    left.append(canvas, diagnostics, warnings);
    if (linkedSize.unresolved.length) left.append(el("div", "font-size:11px;color:#f3be83;margin-top:4px",
        `Unable to preview computed input(s): ${linkedSize.unresolved.join(", ")}. Editor uses stored widget defaults for those values; Python will use actual runtime values and rescale the masks.`));
    else if (linkedSize.resolvedFromLinks) left.append(el("div", "font-size:11px;color:#a5dfb6;margin-top:4px",
        "Resolution synchronized from connected graph controls at editor opening."));
    const right = el("div", "flex:1 1 260px;max-width:290px;min-width:235px");
    right.append(el("div", "font-weight:700;margin-bottom:7px", "Regions · up to 3"));
    const list = el("div", "display:flex;flex-direction:column;gap:4px;margin-bottom:6px;max-width:100%;min-width:0");
    const rowButtons = el("div", "display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px");
    const addBtn = el("button", css.button, "+ Add");
    const removeBtn = el("button", css.button, "Remove");
    for (const b of [addBtn, removeBtn]) b.type = "button";
    rowButtons.append(addBtn, removeBtn);
    const detailTitle = el("div", "font-weight:700;margin-bottom:6px", "Selected region");
    const loraLabel = el("label", "display:flex;flex-direction:column;gap:3px;margin-bottom:6px", "LoRA for this region");
    const loraSelect = el("select", css.input + ";max-width:100%;min-width:0;text-overflow:ellipsis");
    loraLabel.append(loraSelect);
    const strengthLabel = el("label", "display:flex;flex-direction:column;gap:3px;margin-bottom:6px", "LoRA strength");
    const strengthInput = textInput("number");
    strengthInput.min = "-3"; strengthInput.max = "3"; strengthInput.step = "0.05";
    strengthLabel.append(strengthInput);
    const nameLabel = el("label", "display:flex;flex-direction:column;gap:3px;margin-bottom:6px", "Name / label");
    const nameField = textInput("text", "Label only; does not change the LoRA filename"); nameField.maxLength = 64;
    nameLabel.append(nameField);
    const grid = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px");
    const numbers = {};
    for (const [field, label] of [["x", "X"], ["y", "Y"], ["w", "Width"], ["h", "Height"]]) {
        const wrap = el("label", "display:flex;flex-direction:column;gap:3px", label);
        const input = textInput("number"); input.step = "1"; input.min = (field === "w" || field === "h") ? "1" : "0";
        input.max = String(field === "y" || field === "h" ? height : width);
        numbers[field] = input; wrap.append(input); grid.append(wrap);
    }
    const featherTitle = el("div", "font-weight:700;margin-bottom:6px", "Feather · real mask preview");
    const featherRow = el("div", "display:flex;gap:8px;align-items:center;margin-bottom:5px");
    const featherSlider = el("input", "width:100%;flex:1"); featherSlider.type = "range"; featherSlider.min = "0"; featherSlider.max = String(MAX_FEATHER); featherSlider.step = "1";
    const featherNum = textInput("number"); featherNum.min = "0"; featherNum.max = String(MAX_FEATHER);
    featherNum.style.width = "60px"; featherRow.append(featherSlider, featherNum);
    const info = el("div", "font-size:11px;color:#b0bece;margin-bottom:3px",
        "0 = sharp edges; larger values = wider transition. Preview uses the same replicate-padded averaging formula as the Python mask node.");
    right.append(list, rowButtons, detailTitle, loraLabel, strengthLabel, nameLabel, grid, featherTitle, featherRow, info);
    cols.append(left, right); panel.append(cols);

    const fileRow = el("div", "display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:5px");
    fileRow.append(el("span", "", "Reference image:"));
    const imageStatus = el("span", "font-size:11px;color:#bdd2e8;max-width:290px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
        connectedImage.linked ? "Connected image: checking..." : "Optional: choose an image below");
    const reload = el("button", css.button, "Use connected image"); reload.type = "button";
    reload.disabled = !connectedImage.linked;
    const file = el("input", "max-width:100%"); file.type = "file"; file.accept = "image/*";
    const clear = el("button", css.button, "Clear image"); clear.type = "button";
    fileRow.append(file, reload, clear, imageStatus); panel.append(fileRow);
    const footer = el("div", "display:flex;gap:8px;justify-content:flex-end;border-top:1px solid #4e5b6a;margin-top:6px;padding-top:7px");
    const cancel = el("button", css.button, "Cancel"); cancel.type = "button";
    const apply = el("button", css.button + ";background:#246ac0;border-color:#4184d5", "Apply regions"); apply.type = "button";
    footer.append(cancel, apply); panel.append(footer); overlay.append(panel); document.body.appendChild(overlay);

    const getCurrent = () => regions.find(r => r.id === selectedId && r.enabled) || null;
    const others = () => regions.filter(r => r.enabled && r.id !== selectedId);
    function close() {
        if (closed) return;
        closed = true;
        if (raf !== null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(raf);
        referenceLoadId++;
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        document.removeEventListener("keydown", onKeyDown, true);
        overlay.remove();
    }
    function select(id) {
        selectedId = id;
        updateList(); updateFields(); scheduleDraw();
    }
    function updateList() {
        list.replaceChildren();
        for (const r of regions) {
            const b = el("button", css.button + `;text-align:left;border-color:${r.id === selectedId ? regionColor(r.id) : '#566372'};background:${r.id === selectedId ? '#3d4b61' : '#2d3642'};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;width:100%;max-width:100%;min-width:0;padding:5px 7px;display:block`);
            b.type = "button";
            const shortName = pending[r.id].lora === "None" ? "choose LoRA" : pending[r.id].lora.split(/[\/]/).pop();
            b.textContent = `${r.enabled ? '●' : '○'} ${r.id}. ${r.name} — ${shortName}${r.enabled ? '' : ' (inactive)'}`;
            b.title = b.textContent;
            b.style.color = r.enabled ? regionColor(r.id) : "#8e99a7";
            b.addEventListener("click", () => {
                if (!r.enabled) r.enabled = true;
                select(r.id);
            });
            list.append(b);
        }
        addBtn.disabled = !regions.some(r => !r.enabled);
        removeBtn.disabled = !getCurrent();
    }
    function updateFields() {
        const current = getCurrent();
        for (const field of [nameField, ...Object.values(numbers), featherSlider, featherNum, loraSelect, strengthInput]) field.disabled = !current;
        detailTitle.textContent = current ? `Region ${current.id} · MASK ${current.id}` : "Select or add a region";
        if (!current) return;
        const loraWidget = widgets[`lora_${current.id}`];
        const options = loraWidget?.options?.values ?? [pending[current.id].lora];
        const names = Array.isArray(options) ? options : [pending[current.id].lora];
        loraSelect.replaceChildren();
        for (const value of new Set(["None", ...names, pending[current.id].lora])) {
            const option = el("option"); option.value = String(value); option.textContent = String(value);
            loraSelect.append(option);
        }
        loraSelect.value = pending[current.id].lora;
        strengthInput.value = String(pending[current.id].strength);
        nameField.value = current.name;
        for (const key of Object.keys(numbers)) numbers[key].value = String(current[key]);
        featherSlider.value = String(current.feather);
        featherNum.value = String(current.feather);
    }
    function repaint() {
        if (closed) return;
        const visible = regions.filter(r => r.enabled);
        const sx = canvas.width / width, sy = canvas.height / height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#303944"; ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (reference && mode === "overlay") ctx.drawImage(reference, 0, 0, canvas.width, canvas.height);
        else if (mode === "overlay") {
            ctx.strokeStyle = "#ffffff16";
            for (let i = 1; i < 10; i++) {
                ctx.beginPath(); ctx.moveTo(canvas.width * i / 10, 0); ctx.lineTo(canvas.width * i / 10, canvas.height); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, canvas.height * i / 10); ctx.lineTo(canvas.width, canvas.height * i / 10); ctx.stroke();
            }
        }
        const data = renderMaskPixels(visible, width, height, canvas.width, canvas.height, mode, selectedId);
        const pixels = maskCtx.createImageData(canvas.width, canvas.height);
        pixels.data.set(data); maskCtx.putImageData(pixels, 0, 0);
        ctx.drawImage(maskCanvas, 0, 0);
        for (const r of visible) {
            ctx.save(); ctx.strokeStyle = regionColor(r.id); ctx.lineWidth = r.id === selectedId ? 2.6 : 1.6;
            ctx.setLineDash(r.id === selectedId ? [] : [6, 4]);
            ctx.strokeRect(r.x * sx, r.y * sy, r.w * sx, r.h * sy); ctx.setLineDash([]);
            const label = `${r.id}. ${r.name}`;
            ctx.font = "bold 11px sans-serif";
            ctx.fillStyle = "#1d2534de";
            ctx.fillRect(r.x * sx, Math.max(0, r.y * sy - 19), Math.min(canvas.width, ctx.measureText(label).width + 12), 18);
            ctx.fillStyle = regionColor(r.id);
            ctx.fillText(label, r.x * sx + 5, Math.max(12, r.y * sy - 6));
            ctx.restore();
        }
        let overlapCount = 0;
        for (let i = 0; i < visible.length; i++) {
            for (let j = i + 1; j < visible.length; j++) {
                const shared = intersection(visible[i], visible[j]);
                if (!shared) continue;
                overlapCount++;
                if (overlapCheck.checked) {
                    ctx.fillStyle = "#ff744b99";
                    ctx.fillRect(shared.x * sx, shared.y * sy, shared.w * sx, shared.h * sy);
                    ctx.strokeStyle = "#ffd08c";
                    ctx.strokeRect(shared.x * sx, shared.y * sy, shared.w * sx, shared.h * sy);
                }
            }
        }
        const active = getCurrent();
        if (active) {
            const points = [
                [active.x, active.y], [active.x + active.w / 2, active.y], [active.x + active.w, active.y],
                [active.x + active.w, active.y + active.h / 2], [active.x + active.w, active.y + active.h],
                [active.x + active.w / 2, active.y + active.h], [active.x, active.y + active.h],
                [active.x, active.y + active.h / 2],
            ];
            for (const [x, y] of points) {
                ctx.fillStyle = "#f8fafc"; ctx.fillRect(x * sx - 4.5, y * sy - 4.5, 9, 9);
                ctx.strokeStyle = regionColor(active.id); ctx.strokeRect(x * sx - 4.5, y * sy - 4.5, 9, 9);
            }
            diagnostics.textContent = `Selected #${active.id}: X ${active.x}, Y ${active.y}, Width ${active.w}, Height ${active.h}, Feather ${active.feather}px`;
            for (const key of Object.keys(numbers)) if (document.activeElement !== numbers[key]) numbers[key].value = String(active[key]);
            if (document.activeElement !== featherNum) featherNum.value = String(active.feather);
            if (document.activeElement !== featherSlider) featherSlider.value = String(active.feather);
        } else diagnostics.textContent = "No active regions. Click + Add to start.";
        if (activeGuides.guideX !== null || activeGuides.guideY !== null) {
            ctx.strokeStyle = "#ffe3a0"; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
            if (activeGuides.guideX !== null) { ctx.beginPath(); ctx.moveTo(activeGuides.guideX * sx, 0); ctx.lineTo(activeGuides.guideX * sx, canvas.height); ctx.stroke(); }
            if (activeGuides.guideY !== null) { ctx.beginPath(); ctx.moveTo(0, activeGuides.guideY * sy); ctx.lineTo(canvas.width, activeGuides.guideY * sy); ctx.stroke(); }
            ctx.setLineDash([]);
        }
        warnings.style.background = overlapCount ? "#513629" : "#2d4239";
        const missing = visible.filter(r => pending[r.id].lora === "None");
        warnings.textContent = missing.length
            ? `Choose a LoRA for active region(s): ${missing.map(r => r.id).join(", ")}. Generation will reject missing LoRAs.`
            : overlapCount
            ? `${overlapCount} pair(s) of BBoxes overlap. Overlap is allowed, but LoRA identities can mix.`
            : "No geometric overlap. Feather still affects nearby pixels; attention interactions may persist.";
        if (snapFeedback && !missing.length) warnings.textContent += ` ${snapFeedback}`;
        warnings.style.background = missing.length ? "#59452c" : overlapCount ? "#513629" : "#2d4239";
    }
    function scheduleDraw() {
        if (closed || raf !== null) return;
        if (typeof requestAnimationFrame === "function") {
            raf = requestAnimationFrame(() => { raf = null; repaint(); });
        } else repaint();
    }
    function position(ev) {
        const rect = canvas.getBoundingClientRect();
        return { x: clamp(Math.floor((ev.clientX - rect.left) * width / rect.width), 0, width),
            y: clamp(Math.floor((ev.clientY - rect.top) * height / rect.height), 0, height) };
    }
    function hitHandle(point, r) {
        if (!r) return null;
        const b = canvas.getBoundingClientRect();
        const dx = 9 * width / Math.max(1, b.width), dy = 9 * height / Math.max(1, b.height);
        const l = Math.abs(point.x - r.x) <= dx, rr = Math.abs(point.x - r.x - r.w) <= dx;
        const t = Math.abs(point.y - r.y) <= dy, bb = Math.abs(point.y - r.y - r.h) <= dy;
        const horizontal = point.x >= r.x - dx && point.x <= r.x + r.w + dx;
        const vertical = point.y >= r.y - dy && point.y <= r.y + r.h + dy;
        if (l && t) return "nw"; if (rr && t) return "ne";
        if (l && bb) return "sw"; if (rr && bb) return "se";
        if (t && horizontal) return "n"; if (bb && horizontal) return "s";
        if (l && vertical) return "w"; if (rr && vertical) return "e";
        return null;
    }
    const inside = (point, r) => point.x >= r.x && point.x <= r.x + r.w && point.y >= r.y && point.y <= r.y + r.h;
    const cursors = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };
    function options(ev) {
        if (!snapCheck.checked || ev.altKey) return null;
        const b = canvas.getBoundingClientRect();
        return {
            guides: collectGuides(width, height, others(), { canvasEdges: true, boxEdges: true }),
            tolX: clamp(Number(snapTolerance.value) || 10, 1, 30) * width / Math.max(1, b.width),
            tolY: clamp(Number(snapTolerance.value) || 10, 1, 30) * height / Math.max(1, b.height),
        };
    }
    canvas.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        const point = position(ev);
        let r = getCurrent();
        let handle = hitHandle(point, r);
        if (!handle && (!r || !inside(point, r))) {
            const target = [...regions].reverse().find(candidate => candidate.enabled && inside(point, candidate));
            if (target) { select(target.id); r = target; handle = hitHandle(point, r); }
        }
        if (!r) return;
        ev.preventDefault();
        drag = { start: point, original: { ...r }, mode: handle ? "resize" : inside(point, r) ? "move" : "draw", handle };
        if (drag.mode === "draw") Object.assign(r, { x: clamp(point.x, 0, width - 1), y: clamp(point.y, 0, height - 1), w: 1, h: 1 });
        activeGuides = { guideX: null, guideY: null };
        snapFeedback = "";
        canvas.focus(); canvas.setPointerCapture(ev.pointerId); scheduleDraw();
    });
    canvas.addEventListener("pointermove", (ev) => {
        const point = position(ev);
        if (!drag) {
            const r = getCurrent(), handle = hitHandle(point, r);
            canvas.style.cursor = handle ? cursors[handle] : r && inside(point, r) ? "move" : "crosshair";
            return;
        }
        const r = getCurrent(); if (!r) return;
        const opt = options(ev); activeGuides = { guideX: null, guideY: null };
        if (drag.mode === "move") {
            let rect = normalizeRect({ ...drag.original,
                x: clamp(drag.original.x + point.x - drag.start.x, 0, width - drag.original.w),
                y: clamp(drag.original.y + point.y - drag.start.y, 0, height - drag.original.h),
            }, width, height);
            if (opt) ({ rect, ...activeGuides } = snapMove(rect, width, height, opt.guides, opt.tolX, opt.tolY));
            Object.assign(r, rect);
        } else if (drag.mode === "resize") {
            const a = drag.original;
            let l = a.x, rr = a.x + a.w, t = a.y, b = a.y + a.h;
            if (drag.handle.includes("w")) l = clamp(point.x, 0, rr - 1);
            if (drag.handle.includes("e")) rr = clamp(point.x, l + 1, width);
            if (drag.handle.includes("n")) t = clamp(point.y, 0, b - 1);
            if (drag.handle.includes("s")) b = clamp(point.y, t + 1, height);
            let rect = { x: l, y: t, w: rr - l, h: b - t };
            if (opt) ({ rect, ...activeGuides } = snapResize(rect, drag.handle, width, height, opt.guides, opt.tolX, opt.tolY));
            Object.assign(r, rect);
        } else {
            let ex = point.x, ey = point.y;
            if (opt) {
                const hx = opt.guides.xs.filter(g => Math.abs(ex - g.value) <= opt.tolX).sort((a, b) => Math.abs(a.value - ex) - Math.abs(b.value - ex))[0];
                const hy = opt.guides.ys.filter(g => Math.abs(ey - g.value) <= opt.tolY).sort((a, b) => Math.abs(a.value - ey) - Math.abs(b.value - ey))[0];
                if (hx) { ex = hx.value; activeGuides.guideX = ex; }
                if (hy) { ey = hy.value; activeGuides.guideY = ey; }
            }
            Object.assign(r, normalizeRect({ x: Math.min(drag.start.x, ex), y: Math.min(drag.start.y, ey),
                w: Math.abs(ex - drag.start.x), h: Math.abs(ey - drag.start.y) }, width, height));
        }
        if (activeGuides.guideX !== null || activeGuides.guideY !== null) {
            snapFeedback = `Snap applied (${snapTolerance.value} px radius); hold Alt to bypass.`;
        }
        scheduleDraw();
    });
    const release = () => { drag = null; activeGuides = { guideX: null, guideY: null }; updateFields(); scheduleDraw(); };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("lostpointercapture", release);

    function addRegion() {
        const slot = regions.find(r => !r.enabled);
        if (!slot) return;
        // IMPORTANT: choosing the LoRA in the node before pressing Add must
        // not silently replace it with None.
        Object.assign(slot, normalizeRegion({ ...slot, enabled: true }, width, height, slot.id));
        select(slot.id);
    }
    addBtn.addEventListener("click", addRegion);
    removeBtn.addEventListener("click", () => {
        const r = getCurrent(); if (!r) return;
        r.enabled = false;
        pending[r.id].lora = "None";
        pending[r.id].strength = 1.0;
        select(regions.find(candidate => candidate.enabled)?.id ?? null);
    });
    loraSelect.addEventListener("change", () => {
        const r = getCurrent(); if (!r) return;
        pending[r.id].lora = loraSelect.value;
        updateList(); scheduleDraw();
    });
    strengthInput.addEventListener("change", () => {
        const r = getCurrent(); if (!r || !Number.isFinite(Number(strengthInput.value))) return;
        pending[r.id].strength = clamp(Number(strengthInput.value), -3, 3);
        strengthInput.value = String(pending[r.id].strength);
        scheduleDraw();
    });
    nameField.addEventListener("input", () => {
        const r = getCurrent(); if (!r) return;
        r.name = nameField.value.slice(0, 64);
        updateList(); scheduleDraw();
    });
    for (const [key, input] of Object.entries(numbers)) {
        input.addEventListener("change", () => {
            const r = getCurrent(); if (!r || !Number.isFinite(Number(input.value))) return;
            const value = Math.round(Number(input.value));
            const rect = { ...r, [key]: value };
            if (key === "x") rect.x = clamp(value, 0, width - r.w);
            if (key === "y") rect.y = clamp(value, 0, height - r.h);
            Object.assign(r, normalizeRect(rect, width, height));
            updateFields(); scheduleDraw();
        });
    }
    function changeFeather(value) {
        const r = getCurrent(); if (!r || !Number.isFinite(Number(value))) return;
        r.feather = clamp(Math.round(Number(value)), 0, MAX_FEATHER);
        featherNum.value = String(r.feather); featherSlider.value = String(r.feather); scheduleDraw();
    }
    featherSlider.addEventListener("input", () => changeFeather(featherSlider.value));
    featherNum.addEventListener("input", () => changeFeather(featherNum.value));
    modeSelect.addEventListener("change", () => { mode = modeSelect.value; scheduleDraw(); });
    for (const c of [overlapCheck, snapCheck]) c.addEventListener("change", scheduleDraw);
    snapTolerance.addEventListener("input", () => {
        snapTolerance.value = String(clamp(Math.round(Number(snapTolerance.value) || 10), 1, 30));
        snapReadout.textContent = `${snapTolerance.value} px`;
        scheduleDraw();
    });
    function loadReference(url, label, fallback = null) {
        const request = ++referenceLoadId;
        const img = new Image();
        imageStatus.textContent = `Loading: ${label}`;
        img.onload = () => {
            if (closed || request !== referenceLoadId) return;
            reference = img;
            imageStatus.textContent = label;
            scheduleDraw();
        };
        img.onerror = () => {
            if (closed || request !== referenceLoadId) return;
            if (fallback) { fallback(); return; }
            reference = null;
            imageStatus.textContent = "No reference available yet — run the workflow once or select a file";
            scheduleDraw();
        };
        img.src = url;
    }
    function showConnected() {
        if (!connectedImage.linked) return;
        file.value = "";
        if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
        reference = null;
        // ComfyUI's native Load Image file is accessible before sampling.
        // Other IMAGE producers are cached after this node executes.
        const cached = cachedReferencePath(node);
        const cacheFallback = () => {
            if (!cached) { imageStatus.textContent = "Run the workflow once to preview this IMAGE input"; return; }
            loadReference(api.apiURL(cached) + `?v=${Date.now()}`,
                "Connected IMAGE — most recent workflow execution");
        };
        if (connectedImage.kind === "load-image") {
            loadReference(api.apiURL(connectedImage.path),
                `Connected ${connectedImage.label} — automatic preview`, cacheFallback);
        } else cacheFallback();
    }
    file.addEventListener("change", () => {
        const uploaded = file.files?.[0];
        referenceLoadId++;
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = null;
        reference = null;
        if (!uploaded) { imageStatus.textContent = "No manual file chosen"; scheduleDraw(); return; }
        objectUrl = URL.createObjectURL(uploaded);
        loadReference(objectUrl, `Manual reference: ${uploaded.name}`);
    });
    reload.addEventListener("click", showConnected);
    clear.addEventListener("click", () => {
        referenceLoadId++;
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = null; reference = null; file.value = "";
        imageStatus.textContent = "Reference cleared (boxes remain unchanged)";
        scheduleDraw();
    });
    if (connectedImage.linked) showConnected();
    function onKeyDown(ev) {
        if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); close(); return; }
        if (ev.target?.matches?.("input,textarea,select,[contenteditable]")) return;
        const step = ev.shiftKey ? 10 : 1;
        const delta = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[ev.key];
        const r = getCurrent();
        if (!delta || !r) return;
        ev.preventDefault(); ev.stopPropagation();
        Object.assign(r, normalizeRect({ ...r,
            x: clamp(r.x + delta[0], 0, width - r.w),
            y: clamp(r.y + delta[1], 0, height - r.h) }, width, height));
        scheduleDraw();
    }
    document.addEventListener("keydown", onKeyDown, true);
    overlay.addEventListener("click", ev => { if (ev.target === overlay) close(); });
    cancel.addEventListener("click", close);
    apply.addEventListener("click", () => {
        const value = saveRegions(regions, width, height);
        for (let id = 1; id <= MAX; id++) {
            for (const [name, next] of [[`lora_${id}`, pending[id].lora], [`strength_${id}`, pending[id].strength]]) {
                const widget = widgets[name];
                if (!widget) continue;
                widget.value = next;
                if (typeof widget.callback === "function") widget.callback(next);
            }
        }
        config.value = value;
        if (typeof config.callback === "function") config.callback(value);
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        close();
    });
    updateList(); updateFields(); scheduleDraw();
}

app.registerExtension({
    name: "spatial_lora.multi_lora_editor",
    async nodeCreated(node) {
        if (!isMulti(node)) return;
        if (node.widgets?.some(w => w.name === "Open Spatial Editor")) return;
        // JSON stays serializable as a normal STRING widget; hide it visually.
        // If the frontend extension fails to load, users can still edit it.
        const json = node.widgets?.find(w => w.name === "regions_json");
        if (json) {
            json.type = "hidden";
            json.computeSize = () => [0, -4];
        }
        node.addWidget("button", "Open Spatial Editor", "Edit LoRAs / regions", () => openEditor(node), { serialize: false });
        node.setSize?.([Math.max(node.size?.[0] || 280, 360), Math.max(node.size?.[1] || 100, node.computeSize?.()?.[1] || 120)]);
    },
});
