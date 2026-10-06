"""Disk-level checks: persisted PNG masks reproduce the reported Dice/IoU."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
from PIL import Image

from .artifacts import RunArtifacts, safe_relative
from .evaluate import main, evaluate, parser
from .task_metrics import SegmentationAccumulator, overlap_scores
from . import test_task_metrics as fixtures


class ArtifactTests(unittest.TestCase):
    setUp = fixtures.IntegrationTests.setUp
    args = fixtures.IntegrationTests.args

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('PyTorch unavailable')
        cls.torch = torch

    def run_cli(self, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main([str(self.root/'real'), str(self.root/'synthetic'), '--metrics', 'frozen_seg',
                  '--task-checkpoint', str(self.checkpoint), '--model-classes', 'a',
                  '--results-dir', str(self.root/'results'), '--run-info', 'test experiment', '--quiet', *extra])
        return json.loads(output.getvalue())

    def test_pngs_reproduce_counts_and_scores(self):
        report = self.run_cli('--seg-threshold', '.7', '--batch-size', '4')
        artifacts = report['artifacts']
        root = Path(artifacts['results_dir'])
        folder = Path(artifacts['segmentation'])
        self.assertEqual(folder.name, Path(artifacts['report']).stem)
        self.assertIn('test_experiment', folder.name)
        self.assertEqual(json.loads(Path(artifacts['report']).read_text()), report)
        self.assertIn('--seg-threshold', (folder/'command.txt').read_text())
        state = json.loads((folder/'run.json').read_text())
        self.assertEqual(state['status'], 'complete')
        for domain in ('real', 'synthetic'):
            counts = np.zeros(3, dtype=int)
            self.assertEqual(report['datasets'][domain]['used_images'], 6)
            self.assertEqual(len(report['metrics']['frozen_seg'][domain]['per_image']), 6)
            for row in report['metrics']['frozen_seg'][domain]['per_image']:
                original = root/row['original']['path']
                self.assertEqual(original.read_bytes(), Path(row['input_image']).read_bytes())
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), row['original']['sha256'])
                score = row['classes']['a']
                prediction = np.asarray(Image.open(root/score['files']['prediction'])) > 0
                truth = np.asarray(Image.open(root/score['files']['target'])) > 0
                tp, fp, fn = int((prediction & truth).sum()), int((prediction & ~truth).sum()), int((~prediction & truth).sum())
                self.assertEqual([tp, fp, fn], [score['tp'], score['fp'], score['fn']])
                counts += (tp, fp, fn)
            expected = overlap_scores(*counts)
            for metric in ('dice', 'iou'):
                self.assertEqual(expected[metric], report['metrics']['frozen_seg'][domain]['classes']['a'][metric])
            self.assertEqual(state['saved_mask_pairs'][domain], 6)
            self.assertEqual(state['saved_originals'][domain], 6)
            self.assertEqual(artifacts['saved_originals'][domain], 6)

    def test_original_preserves_16bit_tiff_and_nested_names(self):
        args = self.args()
        args.results_dir = self.root/'results'
        bundle = RunArtifacts(args, ['python', 'evaluate.py'])
        source = self.root/'testing0001.tif'
        values = np.array([[0, 65535], [32768, 1234]], dtype=np.uint16)
        Image.fromarray(values).save(source)
        saved = bundle.save_original('real', 'nested/lucchi_1', source)
        target = bundle.root/saved['path']
        self.assertEqual(target.name, source.name)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        np.testing.assert_array_equal(np.asarray(Image.open(target)), values)
        self.assertTrue(saved['path'].endswith('real/original/nested/testing0001.tif'))

    def test_repeated_runs_preserve_previous(self):
        a = self.run_cli()
        old = Path(a['artifacts']['report']).read_bytes()
        b = self.run_cli()
        self.assertNotEqual(a['artifacts']['run_name'], b['artifacts']['run_name'])
        self.assertEqual(Path(a['artifacts']['report']).read_bytes(), old)

    def test_failed_run_marked_not_success(self):
        with self.assertRaises(SystemExit):
            self.run_cli('--require-real-only')
        states = list((self.root/'results'/'segmentation').glob('*/run.json'))
        self.assertEqual(len(states), 1)
        self.assertEqual(json.loads(states[0].read_text())['status'], 'failed')
        self.assertFalse(list((self.root/'results'/'report').glob('*.json')))

    def test_no_checkpoint_classical_still_saved(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main([str(self.root/'real'), str(self.root/'synthetic'), '--metrics', 'intensity',
                  '--results-dir', str(self.root/'results'), '--quiet'])
        r = json.loads(output.getvalue())
        self.assertTrue(Path(r['artifacts']['report']).is_file())
        self.assertEqual(r['artifacts']['saved_mask_pairs'], {'real': 0, 'synthetic': 0})

    def test_checkpoint_adds_frozen_to_preset(self):
        defaults = parser().parse_args([])
        self.assertIsNone(defaults.max_images)
        self.assertEqual(defaults.batch_size, 16)
        args = self.args()
        args.metrics = None
        with patch('evaluation.evaluate.Encoder', side_effect=AssertionError('excluded')):
            args.exclude_metrics = ['fid', 'kid']
            self.assertEqual(list(evaluate(args)['metrics']), ['frozen_seg'])
        args.metrics = ['intensity']
        self.assertEqual(list(evaluate(args)['metrics']), ['intensity'])
        self.assertFalse(args.require_real_only)

    def test_multiclass_export_is_argmax_not_threshold(self):
        args = self.args()
        args.results_dir = self.root/'results'
        bundle = RunArtifacts(args, ['python', 'evaluate.py'])
        prob = np.array([[[.35, .4]], [[.4, .35]], [[.25, .25]]])
        masks = {'a': np.array([[True, False]]), 'b': np.array([[False, False]])}
        accumulator = SegmentationAccumulator(['a', 'b'], 0)
        accumulator.add('nested/img.foo', prob, masks, 'multiclass', ['background', 'a', 'b'], .9, 'g',
                        lambda key, name, pred, target: bundle.save_masks('real', key, name, pred, target))
        files = accumulator.rows[0]['classes']['a']['files']
        saved = np.asarray(Image.open(bundle.root/files['prediction'])) > 0
        np.testing.assert_array_equal(saved, masks['a'])
        self.assertTrue(files['prediction'].endswith('img.foo.png'))
        with self.assertRaises(ValueError):
            safe_relative('../escape')


if __name__ == '__main__':
    unittest.main()
