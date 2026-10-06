"""Optional adapter checks with a tiny random CLIP, never downloaded/evaluation weights."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from .embeddings import Encoder
from .evaluate import parser

HAS_TORCH_CLIP = (importlib.util.find_spec('torch') is not None and
                  importlib.util.find_spec('transformers') is not None)


@unittest.skipUnless(HAS_TORCH_CLIP, 'optional torch/transformers dependencies absent')
class EncoderAdapterTests(unittest.TestCase):
    def test_clip_preprocessing_batching_and_unit_norm(self):
        import torch
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for i, shape in enumerate(((31, 45), (52, 27), (16, 16))):
                path = Path(directory) / f'{i}.png'
                Image.fromarray(np.full(shape, i * 80, dtype=np.uint8)).save(path)
                paths[str(i)] = path
            args = parser().parse_args(['a', 'b', '--device', 'cpu', '--batch-size', '2', '--no-cache'])
            encoder = Encoder('clip', args)
            encoder.model = CLIPVisionModelWithProjection(CLIPVisionConfig(
                hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                num_attention_heads=2, image_size=336, patch_size=112, projection_dim=768)).eval()
            old_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(2)
                with patch.object(encoder, 'load'):
                    features, shapes, _ = encoder.features(paths, lambda _: None)
            finally:
                torch.set_num_threads(old_threads)
            self.assertEqual(features.shape, (3, 768))
            np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1., atol=1e-6)
            np.testing.assert_array_equal(shapes, [[31, 45], [52, 27], [16, 16]])
            self.assertTrue(np.isfinite(features).all())


if __name__ == '__main__':
    unittest.main()
