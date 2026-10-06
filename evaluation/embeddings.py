"""Lazy fixed pretrained encoders and content-addressed feature cache."""
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .data import read_intensity

CLIP_MODEL = 'openai/clip-vit-large-patch14-336'
# Pin a revision rather than accepting silently changing remote weights.
CLIP_REVISION = 'ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
FID_WEIGHTS = 'pt_inception-2015-12-05-6726825d.pth'


class EncoderUnavailable(RuntimeError):
    """Missing packages, device, or pretrained weights (never a data error)."""


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return 'not_installed'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(images, specification):
    return {'cache_schema': 1, 'encoder': specification,
            'files': [{'path': str(path.resolve()), 'sha256': sha256_file(path)} for path in images.values()]}


def cached_features(images, specification, cache_dir, compute, progress):
    """compute returns features and original shapes; no pickle is ever loaded."""
    record = manifest(images, specification) if cache_dir is not None else None
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() if record else None
    target = Path(cache_dir) / (key + '.npz') if key else None
    if target is not None and target.exists():
        try:
            with np.load(target, allow_pickle=False) as cached:
                features = cached['features']
                shapes = cached['shapes']
                valid = (json.loads(str(cached['manifest'])) == record and
                         features.shape == (len(images), specification['dimensions']) and
                         shapes.shape == (len(images), 2) and np.isfinite(features).all())
                if not valid:
                    raise ValueError('Invalid cache contents')
            progress(f"{specification['name']}: feature cache hit")
            return features, shapes, {'hit': True, 'path': str(target), 'key': key}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            progress(f'Ignoring corrupt feature cache: {target}')
    features, shapes = compute()
    if features.shape != (len(images), specification['dimensions']) or not np.isfinite(features).all():
        raise ValueError('Encoder returned invalid features')
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replacement permits concurrent readers and interrupted evaluations.
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.npz', delete=False) as stream:
                name = stream.name
                np.savez_compressed(stream, features=features, shapes=shapes,
                                    manifest=json.dumps(record, sort_keys=True))
            os.replace(name, target)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)
    return features, shapes, {'hit': False, 'path': str(target) if target else None, 'key': key}


class Encoder:
    def __init__(self, name, args):
        self.name, self.args, self.model = name, args, None
        try:
            import torch
        except (ImportError, RuntimeError) as error:
            raise EncoderUnavailable(f'PyTorch unavailable: {error}. Install evaluation/requirements.txt') from error
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise EncoderUnavailable('CUDA requested but unavailable; use --device cpu')
        dependencies = ['torch', 'torchvision', 'pytorch-fid'] if name == 'inception' else ['torch', 'transformers']
        for dependency in dependencies:
            if package_version(dependency) == 'not_installed':
                raise EncoderUnavailable(f'{dependency} missing; install evaluation/requirements.txt')
        self.specification = {
            'name': name, 'dimensions': 2048 if name == 'inception' else 768,
            'checkpoint': FID_WEIGHTS if name == 'inception' else CLIP_MODEL,
            'revision': None if name == 'inception' else CLIP_REVISION,
            'preprocess': ('gray_repeat_rgb;float32;bilinear299_align_corners_false;range_minus1_plus1'
                           if name == 'inception' else
                           'gray_repeat_rgb;float32;bicubic336_antialias_no_crop;clip_mean_std;l2_normalized_projection'),
            'implementation': 1, 'intensity_max': args.intensity_max, 'device': self.device,
            'packages': {name: package_version(name) for name in dependencies},
        }

    def load(self):
        if self.model is not None:
            return
        try:
            if self.name == 'inception':
                from pytorch_fid.inception import InceptionV3
                checkpoint = Path(self.torch.hub.get_dir()) / 'checkpoints' / FID_WEIGHTS
                if self.args.offline and not checkpoint.exists():
                    raise EncoderUnavailable(f'Offline: FID weights missing at {checkpoint}')
                self.model = InceptionV3([3], resize_input=False, normalize_input=True,
                                         use_fid_inception=True).eval().to(self.device)
            else:
                from transformers import CLIPVisionModelWithProjection
                self.model = CLIPVisionModelWithProjection.from_pretrained(
                    CLIP_MODEL, revision=CLIP_REVISION, local_files_only=self.args.offline
                ).eval().to(self.device)
            self.model.requires_grad_(False)
        except EncoderUnavailable:
            raise
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise EncoderUnavailable(f'{self.name} weights/model unavailable: {error}') from error

    def compute(self, images, progress):
        self.load()
        torch = self.torch
        rows, shapes, pending = [], [], []
        def flush():
            if not pending:
                return
            batch = torch.cat(pending).to(self.device)
            with torch.inference_mode():
                if self.name == 'inception':
                    values = self.model(batch)[0].flatten(1)
                else:
                    values = torch.nn.functional.normalize(self.model(pixel_values=batch).image_embeds, dim=-1)
                rows.append(values.cpu().numpy().astype(np.float32))
            pending.clear()
        for number, path in enumerate(images.values(), 1):
            array = read_intensity(path, self.args.intensity_max)
            shapes.append(array.shape)
            tensor = torch.from_numpy(array.astype(np.float32))[None, None].repeat(1, 3, 1, 1)
            if self.name == 'inception':
                tensor = torch.nn.functional.interpolate(tensor, size=(299, 299), mode='bilinear', align_corners=False)
            else:
                tensor = torch.nn.functional.interpolate(tensor, size=(336, 336), mode='bicubic',
                                                          align_corners=False, antialias=True)
                mean = tensor.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
                std = tensor.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
                tensor = (tensor - mean) / std
            pending.append(tensor)
            if len(pending) == self.args.batch_size:
                flush()
                progress(f'{self.name}: {number}/{len(images)} images encoded')
        flush()
        progress(f'{self.name}: {len(images)}/{len(images)} images encoded')
        return np.concatenate(rows), np.asarray(shapes, dtype=np.int64)

    def features(self, images, progress):
        return cached_features(images, self.specification, None if self.args.no_cache else self.args.cache_dir,
                               lambda: self.compute(images, progress), progress)
