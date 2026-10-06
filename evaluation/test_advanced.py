"""Numerical references, cache invalidation, and preset integration; no weights required."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist

from .advanced import PatchReservoir, descriptors, compare_descriptors
from .distances import fid, kid, polynomial_mmd, prdc, rbf_mmd, sliced_wasserstein, vendi
from .embeddings import cached_features, EncoderUnavailable
from .evaluate import MODES, evaluate, main, parser


class DistanceTests(unittest.TestCase):
    def test_fid_against_covariance_reference(self):
        for n, m, d in [(3, 4, 10), (12, 15, 5)]:
            rng = np.random.default_rng(1)
            a, b = rng.normal(size=(n, d)), rng.normal(1, 2, size=(m, d))
            ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
            expected = np.sum((a.mean(0) - b.mean(0)) ** 2) + np.trace(ca + cb - 2 * sqrtm(ca @ cb)).real
            self.assertAlmostEqual(fid(a, b)['value'], expected, places=5)
            self.assertAlmostEqual(fid(a, a)['value'], 0, places=10)
            self.assertAlmostEqual(fid(a, a + 2)['value'], 4 * d, places=8)

    def test_fid_degenerate_and_single_sample(self):
        self.assertEqual(fid(np.ones((2, 5)), np.zeros((3, 5)))['value'], 5)
        self.assertEqual(fid(np.ones((1, 5)), np.ones((2, 5)))['status'], 'insufficient_samples')

    def test_kid_matches_explicit_unbiased_sums(self):
        x, y = np.array([[0., 1.], [1., 2.], [2., 4.]]), np.array([[2., 1.], [1., 1.], [0., 3.]])
        kernel = lambda a, b: (a.dot(b) / 2 + 1) ** 3
        expected = sum(kernel(a, b) for i, a in enumerate(x) for j, b in enumerate(x) if i != j) / 6
        expected += sum(kernel(a, b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j) / 6
        expected -= 2 * sum(kernel(a, b) for a in x for b in y) / 9
        self.assertAlmostEqual(polynomial_mmd(x, y), expected)
        self.assertAlmostEqual(kid(x, y, 3, 4)['mean'], expected)
        self.assertLess(polynomial_mmd(x, x), 0)  # Unbiased estimates need not be nonnegative.

    def test_cmmd_against_dense_v_statistic(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(7, 4)), rng.normal(size=(9, 4))
        k = lambda a, b: np.exp(-cdist(a, b, 'sqeuclidean') / 200).mean()
        expected = 1000 * (k(x, x) + k(y, y) - 2 * k(x, y))
        self.assertAlmostEqual(rbf_mmd(x, y, scale=1000, block=3)['value'], expected, places=10)
        self.assertAlmostEqual(rbf_mmd(x, x)['value'], 0, places=12)

    def test_prdc_matches_dense_reference(self):
        rng = np.random.default_rng(22)
        x, y, k = rng.normal(size=(10, 5)), rng.normal(1, 2, size=(13, 5)), 3
        rx = np.sort(cdist(x, x), axis=1)[:, k]
        ry = np.sort(cdist(y, y), axis=1)[:, k]
        distance = cdist(x, y)
        expected = {'precision': (distance < rx[:, None]).any(0).mean(),
                    'recall': (distance < ry[None, :]).any(1).mean(),
                    'density': (distance < rx[:, None]).sum(0).mean() / k,
                    'coverage': (distance.min(1) < rx).mean()}
        actual = prdc(x, y, k, block=4)
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])
        self.assertEqual(prdc(x[:3], y, k)['status'], 'insufficient_samples')

    def test_vendi_extremes(self):
        self.assertAlmostEqual(vendi(np.eye(4))['value'], 4)
        self.assertAlmostEqual(vendi(np.ones((4, 4)))['value'], 1)
        self.assertIn('status', vendi(np.zeros((2, 4))))

    def test_swd_identity_and_shift(self):
        x = np.arange(12).reshape(-1, 1)
        self.assertEqual(sliced_wasserstein(x, x)['value'], 0)
        self.assertAlmostEqual(sliced_wasserstein(x, x + 3)['value'], 3)


class CacheTests(unittest.TestCase):
    def test_hits_content_change_and_configuration_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'image.png'
            source.write_bytes(b'abc')
            images = {'sample': source}
            spec = {'name': 'test', 'dimensions': 2, 'revision': 'a'}
            calls = []
            def compute():
                calls.append(1)
                return np.ones((1, 2), dtype=np.float32), np.array([[8, 8]])
            call = lambda: cached_features(images, spec, root / 'cache', compute, lambda _: None)
            self.assertFalse(call()[2]['hit'])
            self.assertTrue(call()[2]['hit'])
            stat = source.stat()
            source.write_bytes(b'def')  # Same size and mtime; content hash must invalidate.
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertFalse(call()[2]['hit'])
            spec['revision'] = 'b'
            self.assertFalse(call()[2]['hit'])
            target = Path(call()[2]['path'])
            target.write_bytes(b'corrupt')
            self.assertFalse(call()[2]['hit'])
            self.assertEqual(len(calls), 4)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.real, self.fake = self.root / 'real', self.root / 'fake'
        rng = np.random.default_rng(3)
        for i in range(3):
            image = rng.integers(0, 256, (40, 40), dtype=np.uint8)
            mask = np.zeros_like(image)
            mask[5:9, 5:9] = 255
            mask[20:24, 20:24] = 255
            for root in (self.real, self.fake):
                for folder, value in [('original', image), ('class1', mask), ('class2', np.roll(mask, 5, axis=1))]:
                    (root / folder).mkdir(parents=True, exist_ok=True)
                    Image.fromarray(value).save(root / folder / f'{i}.png')

    def args(self, *extra):
        return parser().parse_args([str(self.real), str(self.fake), '--quiet', *extra])

    def test_default_and_nested_modes(self):
        self.assertEqual(self.args().mode, 'basic')
        self.assertEqual(MODES['basic'], ('fid', 'kid'))
        self.assertTrue(set(MODES['basic']) < set(MODES['medium']) < set(MODES['full']))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--list-metrics'])
        self.assertEqual(json.loads(output.getvalue())['basic'], ['fid', 'kid'])

    def test_all_modes_with_unavailable_weights_are_explicit(self):
        with patch('evaluation.evaluate.Encoder', side_effect=EncoderUnavailable('offline test')):
            for mode in MODES:
                report = evaluate(self.args('--mode', mode, '--skip-unavailable'))
                self.assertEqual(list(report['metrics']), list(MODES[mode]))
                self.assertEqual(report['status'], 'partial')
                self.assertIsNone(report['metrics']['fid']['value'])
                json.dumps(report, allow_nan=False)
            with self.assertRaises(EncoderUnavailable):
                evaluate(self.args())

    def test_metric_override_does_not_load_models(self):
        with patch('evaluation.evaluate.Encoder', side_effect=AssertionError('must not load')):
            report = evaluate(self.args('--mode', 'full', '--metrics', 'intensity', 'spatial', 'joint_mmd', 'swd'))
        self.assertEqual(set(report['metrics']), {'intensity', 'spatial', 'joint_mmd', 'swd'})
        self.assertAlmostEqual(report['metrics']['joint_mmd']['value'], 0)
        self.assertAlmostEqual(report['metrics']['spatial']['within_class_nearest_centroid_px']['class1']['wasserstein'], 0)
        self.assertAlmostEqual(report['metrics']['swd']['scales']['1']['value'], 0)

    def test_spatial_detects_relocation(self):
        for path in (self.fake / 'class1').glob('*.png'):
            mask = np.zeros((40, 40), dtype=np.uint8)
            mask[5:9, 5:9] = 255
            mask[10:14, 10:14] = 255
            Image.fromarray(mask).save(path)
        report = evaluate(self.args('--metrics', 'spatial'))
        distance = report['metrics']['spatial']['within_class_nearest_centroid_px']['class1']['wasserstein']
        self.assertGreater(distance, 10)

    def test_joint_missing_mask_is_not_imputed(self):
        (self.fake / 'class1' / '0.png').unlink()
        report = evaluate(self.args('--metrics', 'joint_mmd', '--allow-missing-masks'))
        self.assertEqual(report['metrics']['joint_mmd']['synthetic_count'], 2)
        self.assertEqual(report['datasets']['synthetic']['joint_mmd_image_keys'], ['1', '2'])

    def test_bad_parameters_fail(self):
        for option in [('--kid-subset-size', '1'), ('--seed', '-1'), ('--batch-size', '0'), ('--device', 'nonsense')]:
            with self.assertRaises(ValueError):
                evaluate(self.args(*option))

    def test_reservoir_is_bounded_reproducible(self):
        a, b = PatchReservoir(3, 1, 1), PatchReservoir(3, 1, 1)
        for i in range(100):
            a.add([i]); b.add([i])
        np.testing.assert_array_equal(a.values(), b.values())
        self.assertEqual(a.values().shape, (3, 1))
        self.assertGreater(a.values().max(), 10)


if __name__ == '__main__':
    unittest.main()
