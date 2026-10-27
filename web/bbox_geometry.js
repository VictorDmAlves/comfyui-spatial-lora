// Pure geometry helpers for Spatial LoRA's v0.4 editor.
// Kept separate from the UI so that snapping and overlap are easy to test.

export const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

export function normalizeRect(rect, width, height) {
    const w = Math.max(1, Math.round(width));
    const h = Math.max(1, Math.round(height));
    const x = clamp(Math.round(Number(rect.x) || 0), 0, w - 1);
    const y = clamp(Math.round(Number(rect.y) || 0), 0, h - 1);
    return {
        x, y,
        w: clamp(Math.round(Number(rect.w) || 1), 1, w - x),
        h: clamp(Math.round(Number(rect.h) || 1), 1, h - y),
    };
}

export function scaleRect(rect, fromWidth, fromHeight, toWidth, toHeight) {
    if (fromWidth <= 0 || fromHeight <= 0 || toWidth <= 0 || toHeight <= 0) {
        throw new RangeError("Canvas dimensions must be positive");
    }
    return normalizeRect({
        x: rect.x * toWidth / fromWidth,
        y: rect.y * toHeight / fromHeight,
        w: rect.w * toWidth / fromWidth,
        h: rect.h * toHeight / fromHeight,
    }, toWidth, toHeight);
}

export function intersection(a, b) {
    const left = Math.max(a.x, b.x);
    const top = Math.max(a.y, b.y);
    const right = Math.min(a.x + a.w, b.x + b.w);
    const bottom = Math.min(a.y + a.h, b.y + b.h);
    return right > left && bottom > top
        ? { x: left, y: top, w: right - left, h: bottom - top }
        : null;
}

export function overlapDetails(rect, boxes) {
    return boxes.flatMap((box) => {
        const overlap = intersection(rect, box.rect);
        if (!overlap) return [];
        const area = overlap.w * overlap.h;
        return [{
            ...box,
            intersection: overlap,
            area,
            percentOfSmaller: 100 * area / Math.min(rect.w * rect.h, box.rect.w * box.rect.h),
        }];
    });
}

// Guides are in output-image coordinates. Use only boxes sharing the exact
// same coordinate system for snapping; mismatched boxes can be previewed.
export function collectGuides(width, height, boxes, { canvasEdges = true, boxEdges = true } = {}) {
    const xs = [], ys = [];
    const add = (array, value, source) => array.push({ value, source });
    if (canvasEdges) {
        add(xs, 0, "canvas"); add(xs, width, "canvas");
        add(ys, 0, "canvas"); add(ys, height, "canvas");
    }
    if (boxEdges) {
        for (const box of boxes) {
            for (const n of [box.x, box.x + box.w, box.x + box.w / 2]) add(xs, n, "box");
            for (const n of [box.y, box.y + box.h, box.y + box.h / 2]) add(ys, n, "box");
        }
    }
    return { xs, ys };
}

function closestGuide(points, guides, tolerance) {
    let best = null;
    for (const point of points) {
        for (const guide of guides) {
            const delta = guide.value - point;
            if (Math.abs(delta) <= tolerance && (!best || Math.abs(delta) < Math.abs(best.delta))) {
                best = { delta, value: guide.value, source: guide.source };
            }
        }
    }
    return best;
}

export function snapMove(rect, width, height, guides, tolX, tolY) {
    const box = normalizeRect(rect, width, height);
    const sx = closestGuide([box.x, box.x + box.w, box.x + box.w / 2], guides.xs, tolX);
    const sy = closestGuide([box.y, box.y + box.h, box.y + box.h / 2], guides.ys, tolY);
    const result = normalizeRect({
        ...box,
        x: clamp(box.x + (sx?.delta || 0), 0, width - box.w),
        y: clamp(box.y + (sy?.delta || 0), 0, height - box.h),
    }, width, height);
    // Only draw a guide if the constrained position really reached it.
    return {
        rect: result,
        guideX: sx && Math.abs((result.x - box.x) - sx.delta) < 0.51 ? sx.value : null,
        guideY: sy && Math.abs((result.y - box.y) - sy.delta) < 0.51 ? sy.value : null,
    };
}

export function snapResize(rect, handle, width, height, guides, tolX, tolY) {
    if (!/^(n|s|e|w|nw|ne|sw|se)$/.test(handle)) {
        throw new TypeError("Invalid resize handle");
    }
    const raw = normalizeRect(rect, width, height);
    let left = raw.x, right = raw.x + raw.w;
    let top = raw.y, bottom = raw.y + raw.h;
    let gx = null, gy = null;
    if (handle.includes("w")) {
        const hit = closestGuide([left], guides.xs, tolX);
        if (hit && hit.value < right) { left = hit.value; gx = hit.value; }
    }
    if (handle.includes("e")) {
        const hit = closestGuide([right], guides.xs, tolX);
        if (hit && hit.value > left) { right = hit.value; gx = hit.value; }
    }
    if (handle.includes("n")) {
        const hit = closestGuide([top], guides.ys, tolY);
        if (hit && hit.value < bottom) { top = hit.value; gy = hit.value; }
    }
    if (handle.includes("s")) {
        const hit = closestGuide([bottom], guides.ys, tolY);
        if (hit && hit.value > top) { bottom = hit.value; gy = hit.value; }
    }
    left = clamp(left, 0, width - 1); right = clamp(right, left + 1, width);
    top = clamp(top, 0, height - 1); bottom = clamp(bottom, top + 1, height);
    return { rect: { x: left, y: top, w: right - left, h: bottom - top }, guideX: gx, guideY: gy };
}
