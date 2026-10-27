// Exact CPU mask preview for Z-Image Spatial LoRA's replicate-padded
// box-blur feather. No network, GPU, model inference, or ComfyUI dependency.
export const SLOT_COLORS = [
    [82, 168, 255],  // blue
    [94, 232, 145],  // green
    [255, 173, 71],  // amber
];

export function axisCoverage(pixel, start, extent, dimension, feather) {
    if (extent <= 0) return 0;
    if (!feather) return (pixel >= start && pixel < start + extent) ? 1 : 0;
    const low = pixel - feather;
    const high = pixel + feather;
    const insideLow = Math.max(0, low, start);
    const insideHigh = Math.min(dimension - 1, high, start + extent - 1);
    let count = Math.max(0, insideHigh - insideLow + 1);
    // F.pad(mode='replicate') duplicates the edge pixel when outside canvas.
    if (low < 0 && start === 0) count += -low;
    if (high >= dimension && start + extent >= dimension) count += high - (dimension - 1);
    return count / (2 * feather + 1);
}

export function maskValueAt(region, x, y, width, height) {
    if (!region?.enabled) return 0;
    const feather = Math.max(0, Math.round(region.feather || 0));
    return axisCoverage(x, region.x, region.w, width, feather)
        * axisCoverage(y, region.y, region.h, height, feather);
}

// Pixels are sampled at the centers of scaled preview pixels. At native size
// they match the masks emitted by the Python node, up to torch FP rounding.
export function renderMaskPixels(regions, width, height, outWidth, outHeight, mode = 'overlay', activeId = 1) {
    if (!['overlay', 'mask', 'heatmap'].includes(mode)) throw new TypeError('Invalid preview mode');
    const out = new Uint8ClampedArray(outWidth * outHeight * 4);
    const active = regions.filter(r => r.enabled && (mode !== 'mask' || r.id === activeId));
    const xWeights = new Map(), yWeights = new Map();
    for (const region of active) {
        const cx = new Float32Array(outWidth), cy = new Float32Array(outHeight);
        for (let x = 0; x < outWidth; x++) {
            const pixel = Math.min(width - 1, Math.floor((x + .5) * width / outWidth));
            cx[x] = axisCoverage(pixel, region.x, region.w, width, region.feather);
        }
        for (let y = 0; y < outHeight; y++) {
            const pixel = Math.min(height - 1, Math.floor((y + .5) * height / outHeight));
            cy[y] = axisCoverage(pixel, region.y, region.h, height, region.feather);
        }
        xWeights.set(region.id, cx); yWeights.set(region.id, cy);
    }
    for (let y = 0; y < outHeight; y++) {
        for (let x = 0; x < outWidth; x++) {
            const i = 4 * (y * outWidth + x);
            let sum = 0, red = 0, green = 0, blue = 0;
            for (const r of active) {
                const weight = xWeights.get(r.id)[x] * yWeights.get(r.id)[y];
                sum += weight;
                const [cr, cg, cb] = SLOT_COLORS[r.id - 1];
                red += weight * cr; green += weight * cg; blue += weight * cb;
            }
            if (mode === 'mask') {
                const gray = Math.round(Math.min(1, sum) * 255);
                out[i] = gray; out[i + 1] = gray; out[i + 2] = gray; out[i + 3] = 255;
            } else if (mode === 'heatmap') {
                out[i] = Math.min(255, Math.round(red));
                out[i + 1] = Math.min(255, Math.round(green));
                out[i + 2] = Math.min(255, Math.round(blue));
                out[i + 3] = 255;
            } else {
                const denom = Math.max(sum, 1e-9);
                out[i] = Math.round(red / denom);
                out[i + 1] = Math.round(green / denom);
                out[i + 2] = Math.round(blue / denom);
                out[i + 3] = Math.min(210, Math.round(sum * 150));
            }
        }
    }
    return out;
}
