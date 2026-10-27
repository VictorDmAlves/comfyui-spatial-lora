import test from "node:test";
import assert from "node:assert/strict";
import {
    normalizeRect, scaleRect, intersection, overlapDetails,
    collectGuides, snapMove, snapResize,
} from "../web/bbox_geometry.js";

test("rectangle clamps to image dimensions, retaining positive area", () => {
    assert.deepEqual(normalizeRect({ x: 980, y: 970, w: 300, h: 300 }, 1024, 1024),
        { x: 980, y: 970, w: 44, h: 54 });
    assert.deepEqual(normalizeRect({ x: -2, y: 999, w: 0, h: -2 }, 100, 100),
        { x: 0, y: 99, w: 1, h: 1 });
});

test("preview scales different coordinate systems without going out of bounds", () => {
    assert.deepEqual(scaleRect({ x: 50, y: 100, w: 100, h: 300 }, 500, 1000, 1000, 500),
        { x: 100, y: 50, w: 200, h: 150 });
    assert.throws(() => scaleRect({ x: 0, y: 0, w: 1, h: 1 }, 0, 1, 1, 1));
});

test("touching edges do not count as overlap; actual intersect does", () => {
    assert.equal(intersection({ x: 10, y: 10, w: 20, h: 20 }, { x: 30, y: 10, w: 20, h: 20 }), null);
    assert.deepEqual(intersection({ x: 10, y: 10, w: 30, h: 30 }, { x: 20, y: 20, w: 30, h: 20 }),
        { x: 20, y: 20, w: 20, h: 20 });
});

test("overlap alerts report percentage based on smaller rectangle", () => {
    const a = { x: 0, y: 0, w: 100, h: 100 };
    const b = [{ id: 7, rect: { x: 50, y: 0, w: 50, h: 100 }, label: "B" }];
    const result = overlapDetails(a, b);
    assert.equal(result.length, 1);
    assert.equal(result[0].area, 5000);
    assert.equal(result[0].percentOfSmaller, 100);
});

test("move snap removes a slight intersection by aligning adjacent box edges", () => {
    const others = [{ x: 100, y: 100, w: 200, h: 300 }];
    const guides = collectGuides(1024, 1024, others);
    const original = { x: 296, y: 103, w: 200, h: 200 }; // 4px intersection with other
    const snapped = snapMove(original, 1024, 1024, guides, 6, 2);
    assert.deepEqual(snapped.rect, { x: 300, y: 103, w: 200, h: 200 });
    assert.equal(snapped.guideX, 300);
    assert.equal(intersection(snapped.rect, others[0]), null);
});

test("move snap supports canvas boundary and center guide", () => {
    const guides = collectGuides(1000, 800, [{ x: 400, y: 200, w: 200, h: 250 }]);
    assert.equal(snapMove({ x: 4, y: 530, w: 100, h: 100 }, 1000, 800, guides, 6, 3).rect.x, 0);
    assert.equal(snapMove({ x: 451, y: 600, w: 100, h: 100 }, 1000, 800, guides, 3, 2).rect.x, 450);
});

test("resize right edge snaps to left edge of another box", () => {
    const guides = collectGuides(1024, 1024, [{ x: 520, y: 0, w: 200, h: 500 }]);
    const resized = snapResize({ x: 100, y: 200, w: 416, h: 400 }, "e", 1024, 1024, guides, 6, 6);
    assert.deepEqual(resized.rect, { x: 100, y: 200, w: 420, h: 400 });
    assert.equal(resized.guideX, 520);
});

test("resize cannot cross the opposite edge or leave the image", () => {
    const guides = collectGuides(100, 100, [{ x: 0, y: 0, w: 100, h: 100 }]);
    const res = snapResize({ x: 10, y: 20, w: 1, h: 30 }, "w", 100, 100, guides, 20, 5);
    assert.ok(res.rect.w >= 1);
    assert.ok(res.rect.x >= 0);
    assert.ok(res.rect.x + res.rect.w <= 100);
    const second = snapResize({ x: 10, y: 20, w: 89, h: 30 }, "e", 100, 100, guides, 5, 5);
    assert.equal(second.rect.x + second.rect.w, 100);
});

test("no guides means no snapping", () => {
    const guides = collectGuides(100, 100, [], { canvasEdges: false, boxEdges: false });
    assert.deepEqual(snapMove({ x: 10, y: 12, w: 15, h: 16 }, 100, 100, guides, 10, 10).rect,
        { x: 10, y: 12, w: 15, h: 16 });
});
