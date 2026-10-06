"""Run: python -m unittest evaluation.test_evaluation -v"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from .evaluate import CLASSICAL, evaluate, main, parser


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.real, self.synth = self.root / 'real', self.root / 'synth'
        rng = np.random.default_rng(8)
        self.image = rng.integers(0, 256, (48, 48), dtype=np.uint8)
        self.mask = np.zeros((48, 48), dtype=np.uint8)
        self.mask[10:20, 12:24] = 255
        for root in (self.real, self.synth):
            self.write(root, 'original', self.image)
            self.write(root, 'class1', self.mask)

    def write(self, root, folder, array, name='sample.png'):
        directory = root / folder
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(directory / name)

    def run_eval(self, *options):
        if '--metrics' not in options:
            options = (*options, '--metrics', *CLASSICAL)
        args = parser().parse_args([str(self.real), str(self.synth), '--quiet', *options])
        return evaluate(args)

    def test_identity_all_distances_zero(self):
        result = self.run_eval()
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('wasserstein', 'js_distance', 'hellinger', 'total_variation', 'cdf_max_distance'):
                        self.assertAlmostEqual(item, 0, places=12)
                    else:
                        walk(item)
        walk(result['metrics'])
        json.dumps(result, allow_nan=False)

    def test_same_histogram_different_texture(self):
        stripes = np.tile(np.repeat([0, 255], 24), (48, 1)).astype(np.uint8)
        checker = ((np.indices((48, 48)).sum(axis=0) % 2) * 255).astype(np.uint8)
        self.write(self.real, 'original', stripes)
        self.write(self.synth, 'original', checker)
        result = self.run_eval('--metrics', 'intensity', 'lbp', 'glcm')['metrics']
        self.assertEqual(result['intensity']['wasserstein'], 0)
        self.assertGreater(result['lbp']['js_distance'], .1)
        self.assertGreater(result['glcm']['contrast_d1_angle0']['wasserstein'], 1)

    def test_empty_class_is_undefined(self):
        self.write(self.synth, 'class1', np.zeros_like(self.mask))
        result = self.run_eval()['metrics']
        self.assertIn('status', result['class_intensity']['class1'])
        self.assertIsNone(result['morphology']['class1']['area_px']['wasserstein'])
        self.assertGreater(result['coverage']['class1']['wasserstein'], 0)

    def test_missing_mask_strict_and_skip(self):
        self.write(self.real, 'original', self.image, 'second.png')
        with self.assertRaisesRegex(ValueError, 'Missing class1 mask'):
            self.run_eval()
        result = self.run_eval('--allow-missing-masks')
        self.assertEqual(len(result['datasets']['real']['missing_masks']['class1']), 1)
        self.assertEqual(result['metrics']['coverage']['class1']['real']['count'], 1)

    def test_wrong_shape(self):
        self.write(self.synth, 'class1', self.mask[:20])
        with self.assertRaisesRegex(ValueError, 'Mask shape'):
            self.run_eval()

    def test_lucchi_and_class_alias(self):
        for root in (self.real, self.synth):
            (root / 'original' / 'sample.png').unlink()
            (root / 'class1' / 'sample.png').unlink()
        self.write(self.real, 'original', self.image, 'testing0000.png')
        self.write(self.real, 'mitochondrial boundaries', self.mask, 'predict_test0000.png')
        self.write(self.synth, 'original', self.image)
        self.write(self.synth, 'mitochondrial_boundaries', self.mask)
        result = self.run_eval()
        self.assertEqual(result['config']['classes'], ['mitochondrial_boundaries'])
        self.assertEqual(result['metrics']['class_intensity']['mitochondrial_boundaries']['wasserstein'], 0)

    def test_rgba_mask_transparency(self):
        rgba = np.repeat(self.mask[..., None], 4, axis=2)
        rgba[..., 3] = 255
        rgba[0, 0, :3] = 255
        rgba[0, 0, 3] = 0
        self.write(self.real, 'class1', rgba)
        self.assertEqual(self.run_eval()['metrics']['coverage']['class1']['wasserstein'], 0)

    def test_luminance_alpha_image(self):
        la = np.stack((self.image, np.full_like(self.image, 255)), axis=-1)
        self.write(self.real, 'original', la)
        self.assertEqual(self.run_eval()['metrics']['intensity']['wasserstein'], 0)

    def test_intensity_scaling(self):
        self.write(self.real, 'original', self.image.astype(np.uint16) * 257, 'sample.tif')
        (self.real / 'original' / 'sample.png').unlink()
        with self.assertRaisesRegex(ValueError, 'intensity outside'):
            self.run_eval('--metrics', 'intensity')
        self.write(self.synth, 'original', self.image.astype(np.uint16) * 257, 'sample.tif')
        (self.synth / 'original' / 'sample.png').unlink()
        self.assertEqual(self.run_eval('--intensity-max', '65535')['metrics']['intensity']['wasserstein'], 0)

    def test_constant_spectrum(self):
        for root in (self.real, self.synth):
            self.write(root, 'original', np.full_like(self.image, 128))
        self.assertIn('status', self.run_eval('--metrics', 'spectrum')['metrics']['spectrum'])

    def test_seeded_subset_and_json_cli(self):
        for i in range(6):
            for root in (self.real, self.synth):
                self.write(root, 'original', self.image, f'{i}.png')
        a = self.run_eval('--metrics', 'intensity', '--max-images', '3')
        b = self.run_eval('--metrics', 'intensity', '--max-images', '3')
        self.assertEqual(a['datasets']['real']['files'], b['datasets']['real']['files'])
        self.assertEqual(a['datasets']['real']['used_images'], 3)
        output = self.root / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            main([str(self.real), str(self.synth), '--metrics', 'intensity', '--quiet', '--output', str(output),
                  '--results-dir', str(self.root/'results')])
        self.assertEqual(json.loads(stream.getvalue()), json.loads(output.read_text()))

    def test_border_components_excluded(self):
        mask = self.mask.copy()
        mask[:3, :3] = 255
        self.write(self.real, 'class1', mask)
        result = self.run_eval('--metrics', 'morphology')['metrics']['morphology']['class1']
        self.assertEqual(result['excluded_border_components']['real'], 1)
        self.assertEqual(result['area_px']['real']['count'], 1)
        self.assertGreater(result['density_per_mpix']['wasserstein'], 0)


if __name__ == '__main__':
    unittest.main()
