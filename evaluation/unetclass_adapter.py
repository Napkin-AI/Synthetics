"""Adapter for trusted legacy UnetClass Tiny_unet_v3 full-module checkpoints.

Uses the framework's model_block.py and exactly its Tiny_unet_v3 forward graph.
Does not import optional SMP/diffusion/training dependencies.
"""
import importlib.util
import json
import pickle
from pathlib import Path
import types

import torch

from .data import canonical_class
from .task_metrics import digest


class TinyRuntime(torch.nn.Module):
    # nn.Module attributes/submodules are restored by the legacy pickle.
    def forward(self, x):
        x1 = self.init(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.dropout(self.down3(x3))
        x5 = self.dropout2(self.down4(x4))
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        return self.outc(self.up4(y3, x1))


def load(checkpoint, device, config):
    root = Path(config['framework_root'])
    source = root / 'pytorch' / 'src'
    if not source.is_dir():
        raise ValueError('framework_root must point to UnetClass containing pytorch/src')
    training_path = Path(config['training_config'])
    training = json.loads(training_path.read_text())
    if training['model']['type_model'] != 'tiny_unet_v3':
        raise ValueError('Built-in UnetClass adapter supports only tiny_unet_v3')
    if training['model'].get('last_activation') != 'sigmoid_activation':
        raise ValueError('Built-in adapter expects sigmoid multilabel training')
    if training['img_transform_data']['color_mode_img'] != 'gray':
        raise ValueError('Built-in adapter expects grayscale training')
    if training['img_transform_data'].get('normalization_img_fun', 'dev 255') != 'dev 255':
        raise ValueError('Built-in adapter expects dev 255 normalization; supply a custom adapter for other preprocessing')
    spec = importlib.util.spec_from_file_location('_evaluation_unet_blocks', source / 'model_block.py')
    blocks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(blocks)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in ('src.models', 'models') and name == 'Tiny_unet_v3':
                return TinyRuntime
            if module in ('model_block', 'src.model_block'):
                return getattr(blocks, name)
            return super().find_class(module, name)

    pickle_module = types.ModuleType('unetclass_legacy_pickle')
    pickle_module.__dict__.update(vars(pickle))
    pickle_module.Unpickler = Unpickler
    # The user supplies a trusted training artifact. Full-module pickles execute Python.
    model = torch.load(checkpoint, map_location=device, pickle_module=pickle_module, weights_only=False)
    if not isinstance(model, TinyRuntime):
        raise ValueError('Checkpoint is not an UnetClass Tiny_unet_v3 module')
    if model.n_classes != training['train']['num_class'] or model.n_channels != 1:
        raise ValueError('Checkpoint channels disagree with training config')
    feature_layer = config.get('feature_layer', 'down2')
    if feature_layer not in ('init', 'down1', 'down2', 'down3', 'down4'):
        raise ValueError('feature_layer must be init or down1..down4')
    sources = training.get('data_info', [])
    active_sources = [s for s in sources if s.get('proportion_of_dataset', 1) > 0]
    domains = set()
    for row in active_sources:
        name = row.get('dir_img_path', '').lower()
        domains.add('synthetic' if 'synthetic' in name or 'syntetic' in name else
                    'real' if 'original_data' in name or '/slices/' in name else 'unknown')
    provenance = {frozenset({'synthetic'}): 'synthetic_only', frozenset({'real'}): 'real_only',
                  frozenset({'real', 'synthetic'}): 'mixed'}.get(frozenset(domains), 'unknown')

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.feature = None
            self.handle = getattr(model, feature_layer).register_forward_hook(self.capture)
            self.evaluation_metadata = {
                'architecture': 'UnetClass.Tiny_unet_v3', 'feature_layer': feature_layer,
                'expected_model_classes': [canonical_class(c) for c in training['train']['mask_name_label_list']],
                'expected_segmentation_mode': 'multilabel',
                'training_provenance': provenance, 'provenance_evidence': 'adjacent training config; path-based domain classification, not an audited training log',
                'training_sources': sources, 'training_config_sha256': digest(training_path),
                'framework_models_sha256': digest(source/'models.py'),
                'framework_blocks_sha256': digest(source/'model_block.py'),
                'preprocessing': 'gray float32 / intensity_max; no resize; original code divides uint8 by 255',
            }

        def capture(self, module, inputs, output):
            self.feature = output

        def forward(self, images):
            if any(size % 16 for size in images.shape[-2:]):
                raise ValueError('Tiny_unet_v3 adapter expects image dimensions divisible by 16; supply matching training-sized crops')
            logits = self.model(images)
            return {'logits': logits, 'features': self.feature}

    return Adapter()
