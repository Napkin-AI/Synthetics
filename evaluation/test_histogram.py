import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from .histogram import summarize, compare_histograms
from .evaluate import main


class HistogramTests(unittest.TestCase):
    def test_equal_image_weights_and_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for key, value, side in [('black', 0, 2), ('white', 255, 20)]:
                paths[key] = Path(tmp) / (key + '.png')
                Image.fromarray(np.full((side, side), value, dtype=np.uint8)).save(paths[key])
            summary = summarize(paths, 255)
            self.assertEqual(summary['image_count'], 2)
            self.assertEqual(summary['mean_probability'][0], .5)
            self.assertEqual(summary['mean_probability'][-1], .5)
            result = compare_histograms([paths, paths], 255)
            self.assertAlmostEqual(result['overlap'], 1)
            self.assertAlmostEqual(result['distances']['total_variation'], 0)
            result = compare_histograms([{'a': paths['black']}, {'b': paths['white']}], 255)
            self.assertEqual(result['overlap'], 0)

    def test_class_masks_weighting_empty_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = {}, {}
            for i, (value, area) in enumerate([(0, 1), (255, 16), (100, 0)]):
                images[str(i)] = root / f'image{i}.png'
                masks[str(i)] = root / f'mask{i}.png'
                Image.fromarray(np.full((4, 4), value, np.uint8)).save(images[str(i)])
                mask = np.zeros(16, np.uint8)
                mask[:area] = 255
                Image.fromarray(mask.reshape(4, 4)).save(masks[str(i)])
            result = compare_histograms([images, images], 255, masks=[masks, masks], class_name='test')
            self.assertEqual(result['real']['image_count'], 2)
            self.assertEqual(result['real']['empty_mask_images'], 1)
            self.assertEqual(result['real']['mean_probability'][0], .5)
            self.assertEqual(result['real']['mean_probability'][-1], .5)
            with self.assertRaises(ValueError):
                summarize(images, 255, {})
            self.assertEqual(summarize(images, 255, {}, allow_missing=True)['missing_mask_images'], 3)
            empty = {'2': images['2']}
            result = compare_histograms([empty, images], 255, masks=[masks, masks])
            self.assertIsNone(result['overlap'])
            self.assertEqual(result['status'], 'undefined_empty_class')
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(masks['0'])
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                summarize(images, 255, masks)

    def test_cli_plot_and_all_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for domain, count in [('real', 3), ('synthetic', 2)]:
                directory = root / domain / 'original'
                directory.mkdir(parents=True)
                for i in range(count):
                    Image.fromarray(np.full((8, 8), i * 100, dtype=np.uint8)).save(directory / f'{i}.png')
                mask_dir = root / domain / 'mitochondria'
                mask_dir.mkdir()
                for i in range(count):
                    Image.fromarray(np.full((8, 8), 255 if i else 0, np.uint8)).save(mask_dir / f'{i}.png')
            for cap, expected in [(None, (3, 2)), (1, (1, 1))]:
                argv = [str(root / 'real'), str(root / 'synthetic'), '--metrics', 'histogram',
                        '--results-dir', str(root / 'results'), '--quiet']
                if cap:
                    argv += ['--max-images', str(cap)]
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    main(argv)
                report = json.loads(output.getvalue())
                metric = report['metrics']['histogram']
                self.assertEqual(tuple(metric[d]['image_count'] for d in ('real', 'synthetic')), expected)
                self.assertEqual(report['status'], 'complete')
                class_result = metric['classes']['mitochondria']
                for relative in class_result['plots'].values():
                    self.assertTrue((root / 'results' / relative).is_file())
                if cap is None:
                    self.assertEqual(class_result['real']['image_count'], 2)
                    self.assertEqual(class_result['synthetic']['image_count'], 1)
                for extension, relative in metric['plots'].items():
                    path = root / 'results' / relative
                    self.assertTrue(path.is_file())
                    if extension == 'png':
                        with Image.open(path) as image:
                            image.verify()
                self.assertTrue(Path(report['artifacts']['report']).is_file())


if __name__ == '__main__':
    unittest.main()
