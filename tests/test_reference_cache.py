"""Reference preview tests: no change to sampling, bounded in-memory thumbnail."""
import io
import unittest
from PIL import Image
import torch
from test_unified import UnifiedTests
from spatial_unified_test_plugin.reference_cache import (
    cache_reference, get_reference, make_reference_png, _CACHE, _MAX_ENTRIES,
)


class ReferencePreviewTests(unittest.TestCase):
    def tearDown(self):
        _CACHE.clear()

    def test_input_is_optional_and_hidden_id_is_available(self):
        # Reuse real ComfyUI-compatible node class, with a folder_paths mock.
        t = UnifiedTests('test_disabled_region_may_have_no_lora')
        t.setUp()
        try:
            ports = t.main.INPUT_TYPES()
            self.assertEqual(ports['optional']['reference_image'], ('IMAGE',))
            self.assertEqual(ports['hidden']['unique_id'], 'UNIQUE_ID')
        finally:
            t.doCleanups()

    def test_preview_png_downsampling_and_input_not_modified(self):
        image = torch.rand(2, 1200, 1600, 3)
        before = image[0, 0, 0].clone()
        data = make_reference_png(image)
        output = Image.open(io.BytesIO(data))
        self.assertEqual(output.format, 'PNG')
        self.assertEqual(output.size, (1024, 768))
        self.assertTrue(torch.equal(image[0, 0, 0], before))

    def test_clear_and_max_cache_entries(self):
        im = torch.zeros(1, 3, 4, 3)
        for idx in range(_MAX_ENTRIES + 3):
            cache_reference(str(idx), im)
        self.assertEqual(len(_CACHE), _MAX_ENTRIES)
        self.assertIsNone(get_reference('0'))
        self.assertIsNotNone(get_reference(str(_MAX_ENTRIES + 2)))
        cache_reference(str(_MAX_ENTRIES + 2), None)
        self.assertIsNone(get_reference(str(_MAX_ENTRIES + 2)))
        cache_reference('../../private', im)
        self.assertIsNone(get_reference('../../private'))

    def test_reference_never_changes_model_or_mask(self):
        t = UnifiedTests('test_disabled_region_may_have_no_lora')
        t.setUp()
        try:
            args = (t.base,128,64,'auto',
                    __import__('test_unified').layout(),
                    'char0.safetensors',1,'char1.safetensors',1,'None',1)
            normal = t.main.apply(*args)
            with_img = t.main.apply(*args, reference_image=torch.ones(1,12,16,3), unique_id='42')
            self.assertTrue(torch.equal(normal[2], with_img[2]))
            self.assertTrue(torch.equal(normal[3], with_img[3]))
            self.assertTrue(torch.equal(normal[4], with_img[4]))
            self.assertTrue(torch.allclose(normal[0].run(t.x), with_img[0].run(t.x)))
            self.assertEqual(Image.open(io.BytesIO(get_reference('42'))).size,(16,12))
        finally:
            t.doCleanups()

if __name__ == '__main__':
    unittest.main()

class RouteRegistrationTests(unittest.TestCase):
    def test_route_registers_with_promptserver_and_returns_png(self):
        import sys
        import types
        from unittest import mock
        from spatial_unified_test_plugin import reference_cache
        saved = reference_cache._ROUTE_REGISTERED
        routes = []
        class FakeRoutes:
            def get(self, path):
                def inner(fn):
                    routes.append((path, fn))
                    return fn
                return inner
        mock_server = types.ModuleType('server')
        mock_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=FakeRoutes()))
        with mock.patch.dict(sys.modules, {'server': mock_server}):
            reference_cache._ROUTE_REGISTERED = False
            try:
                self.assertTrue(reference_cache.register_reference_route())
                self.assertEqual(routes[0][0], '/spatial_lora/reference/{node_id}')
                self.assertTrue(reference_cache.register_reference_route())
                self.assertEqual(len(routes), 1)
            finally:
                reference_cache._ROUTE_REGISTERED = saved
