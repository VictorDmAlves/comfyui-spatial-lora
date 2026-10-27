import test from 'node:test';
import assert from 'node:assert/strict';
import { axisCoverage, maskValueAt, renderMaskPixels } from '../web/mask_preview.js';

function brute(x, y, region, w, h) {
    if (!region.enabled) return 0;
    let total = 0;
    for (let dy = -region.feather; dy <= region.feather; dy++) {
        for (let dx = -region.feather; dx <= region.feather; dx++) {
            const xx = Math.max(0, Math.min(w - 1, x + dx));
            const yy = Math.max(0, Math.min(h - 1, y + dy));
            if (xx >= region.x && xx < region.x + region.w && yy >= region.y && yy < region.y + region.h) total++;
        }
    }
    return total / (2 * region.feather + 1) ** 2;
}

test('feather analytic value matches brute force replicate padding at boundaries', () => {
    const w = 13, h = 12;
    for (const r of [
        { id: 1, enabled: true, x: 0, y: 0, w: 6, h: 5, feather: 4 },
        { id: 2, enabled: true, x: 7, y: 6, w: 6, h: 6, feather: 5 },
        { id: 3, enabled: true, x: 2, y: 3, w: 5, h: 4, feather: 0 },
        { id: 1, enabled: true, x: 0, y: 0, w, h, feather: 8 },
    ]) {
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            assert.ok(Math.abs(maskValueAt(r, x, y, w, h) - brute(x, y, r, w, h)) < 1e-9,
                `at (${x}, ${y}) feather=${r.feather}`);
        }
    }
});

test('mask preview is grayscale, exact at native pixel resolution', () => {
    const region = { id: 1, enabled: true, x: 0, y: 0, w: 5, h: 4, feather: 2 };
    const img = renderMaskPixels([region], 8, 8, 8, 8, 'mask', 1);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        const i = (8 * y + x) * 4;
        assert.ok(Math.abs(img[i] - 255 * brute(x, y, region, 8, 8)) < 1);
        assert.equal(img[i + 1], img[i]);
        assert.equal(img[i + 2], img[i]);
        assert.equal(img[i + 3], 255);
    }
});

test('overlay produces transparent pixels when no mask is enabled', () => {
    const img = renderMaskPixels([], 128, 128, 32, 32, 'overlay');
    assert.equal(img.length, 32 * 32 * 4);
    assert.equal(img[3], 0);
    const region = { id: 2, enabled: true, x: 1, y: 1, w: 120, h: 120, feather: 8 };
    const colored = renderMaskPixels([region], 128, 128, 32, 32, 'heatmap');
    assert.ok(colored[4 * (16 * 32 + 16) + 1] > 100);
    assert.equal(colored[3], 255);
});

test('rejects invalid modes', () => {
    assert.throws(() => renderMaskPixels([], 8, 8, 8, 8, 'unknown'));
});
