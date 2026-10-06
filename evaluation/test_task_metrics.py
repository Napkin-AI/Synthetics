"""Numerical and integration tests; toy model is not a scientific EM validation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from .c2st import c2st, group_folds
from .em_diagnostics import shape_features, power_profiles, residual_features
from .evaluate import evaluate, parser
from .task_metrics import SegmentationAccumulator, boundary_counts, boundary_f1, overlap_scores


class NumericalTests(unittest.TestCase):
    def test_counts_empty_and_wrong(self):
        self.assertEqual(overlap_scores(3, 1, 2)['dice'], 6/9)
        self.assertEqual(overlap_scores(3, 1, 2)['iou'], .5)
        self.assertIsNone(overlap_scores(0, 0, 0)['dice'])
        self.assertEqual(overlap_scores(0, 2, 0)['dice'], 0.)

    def test_multilabel_overlap_and_empty(self):
        truth = np.zeros((16, 16), dtype=bool)
        truth[3:10, 4:12] = True
        acc = SegmentationAccumulator(['a', 'b', 'empty'], 1)
        masks = {'a': truth, 'b': truth, 'empty': np.zeros_like(truth)}
        acc.add('x', np.stack(list(masks.values())).astype(float), masks, 'multilabel', list(masks), .5, 'slice1')
        r = acc.result()
        self.assertEqual(r['macro_dice'], 1.)
        self.assertEqual(r['classes']['a']['mean_image_brier'], 0.)
        self.assertEqual(r['classes']['empty']['both_empty_images'], 1)
        self.assertIsNone(r['classes']['empty']['dice'])

    def test_boundary_empty_and_tolerance(self):
        a = np.zeros((24, 24), dtype=bool)
        a[5:15, 6:18] = True
        b = np.roll(a, 1, axis=0)
        self.assertEqual(boundary_f1(boundary_counts(a, b, 1)), 1.)
        self.assertLess(boundary_f1(boundary_counts(a, b, 0)), 1.)
        self.assertEqual(boundary_f1(boundary_counts(a, np.zeros_like(a), 2)), 0.)

    def test_group_no_leakage(self):
        groups = np.repeat(['a', 'b', 'c', 'd', 'e', 'f'], 8)
        folds = group_folds(groups, 3, 4)
        for test in folds:
            train = np.setdiff1d(np.arange(len(groups)), test)
            self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertEqual(len(np.unique(np.concatenate(folds))), len(groups))

    def test_c2st_separation_and_null(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(60, 4))
        y = rng.normal(size=(60, 4)) + 8
        groups = np.repeat(np.arange(6), 10).astype(str)
        result = c2st(x, y, groups, groups)
        self.assertGreater(result['auc'], .99)
        self.assertEqual(result['status'], 'ok')
        result = c2st(np.ones((60, 4)), np.ones((60, 4)), groups, groups)
        self.assertEqual(result['auc'], .5)
        self.assertEqual(c2st(x, y, ['one']*60, groups)['status'], 'insufficient_groups')

    def test_spectrum_rotation_and_residual(self):
        x = np.tile(np.sin(np.arange(64)*np.pi/4), (64, 1))
        r, a = power_profiles(x)
        rr, aa = power_profiles(x.T)
        np.testing.assert_allclose(r, rr, atol=1e-12)
        self.assertGreater(np.abs(a-aa).sum(), 1)
        values, _ = residual_features(np.ones((24, 24)), 1.)
        self.assertEqual(values['std'], 0.)
        self.assertIsNone(values['skew'])

    def test_thickness_and_border(self):
        a = np.zeros((48, 48), dtype=bool)
        a[5:40, 20:23] = True
        b = np.zeros_like(a)
        b[5:40, 18:26] = True
        sa, _ = shape_features(a)
        sb, _ = shape_features(b)
        self.assertGreater(np.median(sb['thickness_px']), np.median(sa['thickness_px']))
        a[:8, :8] = True
        _, clipped = shape_features(a)
        self.assertEqual(clipped, 1)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('PyTorch unavailable')
        cls.torch = torch

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for domain in ('real', 'synthetic'):
            for folder in ('original', 'a'):
                (self.root / domain / folder).mkdir(parents=True)
            for i in range(6):
                image = np.zeros((32, 32), dtype=np.uint8)
                image[5:12+i, 8:22] = 255
                for folder in ('original', 'a'):
                    Image.fromarray(image).save(self.root/domain/folder/f'slice{i//2}_patch{i}.png')
        torch = self.torch
        class Toy(torch.nn.Module):
            def forward(self, x):
                return {'logits': (x-.5)*20, 'features': torch.cat((x, x*x, 1-x), dim=1)}
        self.checkpoint = self.root/'model.pt'
        torch.jit.trace(Toy(), torch.zeros((1, 1, 32, 32)), strict=False).save(str(self.checkpoint))

    def args(self, *extra):
        return parser().parse_args([str(self.root/'real'), str(self.root/'synthetic'),
                                   '--metrics', 'em_kid', 'em_fd', 'c2st', 'frozen_seg',
                                   '--task-checkpoint', str(self.checkpoint), '--model-classes', 'a',
                                   '--group-pattern', '^(slice[0-9]+)', '--batch-size', '2', '--kid-subsets', '2', '--quiet', *extra])

    def test_end_to_end_and_json(self):
        r = evaluate(self.args())
        self.assertEqual(r['metrics']['frozen_seg']['real']['macro_dice'], 1.)
        self.assertEqual(r['metrics']['frozen_seg']['synthetic_minus_real_macro']['iou'], 0.)
        self.assertAlmostEqual(r['metrics']['em_fd']['global']['value'], 0., places=10)
        self.assertEqual(r['metrics']['c2st']['classes']['a']['auc'], .5)
        self.assertEqual(r['metrics']['em_kid']['classes']['a']['real_count'], 6)
        json.dumps(r, allow_nan=False)

    def test_missing_group_is_error(self):
        args = self.args()
        args.group_pattern = None
        with self.assertRaisesRegex(ValueError, 'C2ST requires'):
            evaluate(args)

    def test_real_only_guard(self):
        with self.assertRaisesRegex(ValueError, 'provenance is unknown'):
            evaluate(self.args('--require-real-only'))

    def test_channel_order_required(self):
        args = self.args()
        args.model_classes = None
        with self.assertRaisesRegex(ValueError, 'explicit --model-classes'):
            evaluate(args)

    def test_bad_output_channels(self):
        args = self.args()
        args.model_classes = ['a', 'b']
        with self.assertRaisesRegex(ValueError, 'logits must be'):
            evaluate(args)

    def test_absent_class_is_reported(self):
        for path in (self.root/'synthetic'/'a').glob('*.png'):
            Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(path)
        r = evaluate(self.args())
        self.assertEqual(r['metrics']['em_kid']['classes']['a']['status'], 'insufficient_samples')
        self.assertIsNone(r['metrics']['em_kid']['macro_available'])
        self.assertEqual(r['metrics']['frozen_seg']['synthetic']['macro_dice'], 0.)

    def test_multiclass_overlapping_masks_rejected(self):
        for domain in ('real', 'synthetic'):
            (self.root/domain/'b').mkdir()
            for path in (self.root/domain/'a').glob('*.png'):
                (self.root/domain/'b'/path.name).write_bytes(path.read_bytes())
        args = self.args('--metrics', 'em_fd', '--segmentation-mode', 'multiclass', '--model-classes', 'background', 'a', 'b')
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            evaluate(args)

    def test_diagnostics_identity(self):
        r = evaluate(self.args('--metrics', 'shape', 'psd', 'noise'))
        self.assertEqual(r['metrics']['shape']['macro_available'], 0.)
        self.assertEqual(r['metrics']['psd']['radial_log_power_l1'], 0.)
        self.assertEqual(r['metrics']['noise']['features']['std']['wasserstein'], 0.)
        json.dumps(r, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
