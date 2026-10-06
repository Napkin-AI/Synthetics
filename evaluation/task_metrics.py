"""Frozen user-model evaluation; class masks condition feature pooling, never model input."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
from scipy import ndimage

from .c2st import c2st
from .data import canonical_class, read_intensity, read_plane
from .distances import fid, kid
from .embeddings import EncoderUnavailable

TASK_METRICS = ('em_kid', 'em_fd', 'c2st', 'frozen_seg')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def mean_available(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def overlap_scores(tp, fp, fn):
    denominator = 2*tp+fp+fn
    union = tp+fp+fn
    return {'dice': float(2*tp/denominator) if denominator else None,
            'iou': float(tp/union) if union else None,
            'precision': float(tp/(tp+fp)) if tp+fp else None,
            'recall': float(tp/(tp+fn)) if tp+fn else None,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def boundary_counts(pred, truth, tolerance):
    p = pred & ~ndimage.binary_erosion(pred, border_value=0)
    t = truth & ~ndimage.binary_erosion(truth, border_value=0)
    matched_p = int((ndimage.distance_transform_edt(~t)[p] <= tolerance).sum()) if t.any() else 0
    matched_t = int((ndimage.distance_transform_edt(~p)[t] <= tolerance).sum()) if p.any() else 0
    return np.array([matched_p, int(p.sum()), matched_t, int(t.sum())], dtype=np.int64)


def boundary_f1(counts):
    mp, p, mt, t = counts
    if p+t == 0:
        return None
    precision, recall = mp/p if p else 0., mt/t if t else 0.
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.


class SegmentationAccumulator:
    def __init__(self, classes, tolerance):
        self.classes, self.tolerance = classes, tolerance
        self.counts = {c: np.zeros(3, dtype=np.int64) for c in classes}
        self.boundary = {c: np.zeros(4, dtype=np.int64) for c in classes}
        self.rows = []

    def add(self, key, probabilities, masks, mode, channel_names, threshold, group, save_masks=None):
        hard = probabilities >= threshold if mode == 'multilabel' else np.arange(len(probabilities))[:, None, None] == probabilities.argmax(0)
        row = {'key': key, 'group': group, 'classes': {}}
        for name, truth in masks.items():
            p = probabilities[channel_names.index(name)]
            prediction = hard[channel_names.index(name)]
            tp = int(np.count_nonzero(prediction & truth))
            fp, fn = int(np.count_nonzero(prediction & ~truth)), int(np.count_nonzero(~prediction & truth))
            self.counts[name] += (tp, fp, fn)
            boundary = boundary_counts(prediction, truth, self.tolerance)
            self.boundary[name] += boundary
            stats = overlap_scores(tp, fp, fn)
            clipped = np.clip(p, 1e-7, 1-1e-7)
            stats.update(boundary_f1=boundary_f1(boundary), target_pixels=int(truth.sum()),
                         predicted_pixels=int(prediction.sum()), brier=float(np.mean((p-truth)**2)),
                         binary_entropy=float(np.mean(-clipped*np.log(clipped)-(1-clipped)*np.log(1-clipped))),
                         mean_foreground_probability=float(p.mean()))
            row['classes'][name] = stats
            if save_masks is not None:
                stats['files'] = save_masks(key, name, prediction, truth)
        if mode == 'multiclass':
            row['categorical_entropy'] = float(np.mean(-np.sum(probabilities*np.log(np.clip(probabilities, 1e-7, 1)), axis=0)))
        self.rows.append(row)

    def result(self):
        classes = {}
        for name in self.classes:
            rows = [row['classes'][name] for row in self.rows if name in row['classes']]
            score = overlap_scores(*self.counts[name])
            score.update(boundary_f1=boundary_f1(self.boundary[name]), images_evaluated=len(rows),
                         images_with_target=sum(r['target_pixels'] > 0 for r in rows),
                         both_empty_images=sum(r['dice'] is None for r in rows),
                         mean_image_dice=mean_available([r['dice'] for r in rows]),
                         mean_image_iou=mean_available([r['iou'] for r in rows]),
                         mean_image_brier=mean_available([r['brier'] for r in rows]),
                         mean_image_binary_entropy=mean_available([r['binary_entropy'] for r in rows]))
            classes[name] = score
        return {'classes': classes,
                'macro_dice': mean_available([r['dice'] for r in classes.values()]),
                'macro_iou': mean_available([r['iou'] for r in classes.values()]),
                'macro_boundary_f1': mean_available([r['boundary_f1'] for r in classes.values()]),
                'defined_classes': [c for c, r in classes.items() if r['dice'] is not None],
                'per_image': self.rows}


def load_model(args):
    if args.task_checkpoint is None:
        raise EncoderUnavailable('Task metrics need --task-checkpoint and a model adapter (or a TorchScript dictionary model)')
    try:
        import torch
    except ImportError as error:
        raise EncoderUnavailable('Task metrics require PyTorch') from error
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    config = json.loads(args.task_config.read_text()) if args.task_config else {}
    metadata = {'checkpoint': str(args.task_checkpoint.resolve()), 'checkpoint_sha256': digest(args.task_checkpoint),
                'adapter': args.task_adapter or 'torchscript', 'config': config, 'device': device,
                'torch_version': torch.__version__}
    if args.task_adapter:
        module_name, separator, function_name = args.task_adapter.rpartition(':')
        if not separator:
            raise ValueError('--task-adapter must be module:function or /path/adapter.py:function')
        if module_name.endswith('.py'):
            spec = importlib.util.spec_from_file_location('_evaluation_user_adapter', module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        metadata['adapter_sha256'] = digest(module.__file__)
        model = getattr(module, function_name)(args.task_checkpoint, device, config)
    else:
        model = torch.jit.load(str(args.task_checkpoint), map_location=device)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata.update(getattr(model, 'evaluation_metadata', {}))
    metadata['training_provenance'] = metadata.get('training_provenance', config.get('training_provenance', 'unknown'))
    if 'frozen_seg' in (args.metrics or ()) or args.model_classes:
        expected = metadata.get('expected_model_classes')
        if expected and [canonical_class(c) for c in (args.model_classes or [])] != expected:
            raise ValueError(f'--model-classes disagrees with checkpoint training config; expected {expected}')
        if metadata.get('expected_segmentation_mode', args.segmentation_mode) != args.segmentation_mode:
            raise ValueError('--segmentation-mode disagrees with checkpoint training config')
    return model, torch, device, metadata


def source_groups(subsets, args):
    if args.groups_json and args.group_pattern:
        raise ValueError('Choose --groups-json or --group-pattern')
    manifest = json.loads(args.groups_json.read_text()) if args.groups_json else None
    pattern = re.compile(args.group_pattern) if args.group_pattern else None
    if pattern and pattern.groups != 1:
        raise ValueError('--group-pattern needs exactly one capture group for the source slice/volume ID')
    result = []
    for domain, subset in zip(('real', 'synthetic'), subsets):
        groups = {}
        for key in subset:
            if manifest is not None:
                value = manifest.get(domain, {}).get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f'Missing/non-string group for {domain}:{key}')
            elif pattern:
                match = pattern.search(key)
                if not match or not match[1]:
                    raise ValueError(f'--group-pattern did not capture group for {domain}:{key}')
                value = match[1]
            else:
                value = key
            groups[key] = value
        result.append(groups)
    return result


def encode_dataset(model, torch, device, images, indexes, classes, args, active, groups, progress, save_masks=None, save_original=None):
    features = {'global': [], **{c: [] for c in classes}}
    keys = {name: [] for name in features}
    masks_missing, shapes = {}, {}
    channels = [canonical_class(c) for c in (args.model_classes or classes)]
    if len(set(channels)) != len(channels) or not set(classes) <= set(channels):
        raise ValueError('--model-classes must be unique and contain every evaluated class')
    if args.segmentation_mode == 'multiclass' and set(channels) != set(classes) | {'background'}:
        raise ValueError('Multiclass requires --model-classes background plus all evaluated foreground classes')
    accumulator = SegmentationAccumulator(classes, args.boundary_tolerance)
    need_features = bool(set(active) & {'em_kid', 'em_fd', 'c2st'})
    items = list(images.items())
    for offset in range(0, len(items), args.batch_size):
        # Batches preserve shapes; heterogeneous datasets use one forward per shape.
        buckets = {}
        for key, path in items[offset:offset+args.batch_size]:
            im = read_intensity(path, args.intensity_max).astype(np.float32)
            buckets.setdefault(im.shape, []).append((key, path, im))
        for shape, batch in buckets.items():
            size_name = 'x'.join(map(str, shape))
            shapes[size_name] = shapes.get(size_name, 0) + len(batch)
            tensor = torch.from_numpy(np.stack([row[2] for row in batch])[:, None]).to(device)
            with torch.inference_mode():
                outputs = model(tensor)
            if not isinstance(outputs, dict):
                raise ValueError('Task adapter must return a dictionary with logits and/or features')
            if need_features:
                fmap = outputs.get('features')
                if fmap is None or fmap.ndim != 4 or fmap.shape[0] != len(batch) or not torch.isfinite(fmap).all():
                    raise ValueError('features must be finite B x D x Hf x Wf spatial encoder activations')
                if min(fmap.shape[1:]) < 1:
                    raise ValueError('Empty feature map')
                fmap = fmap.detach().float()
            if 'frozen_seg' in active:
                logits = outputs.get('logits')
                expected = (len(batch), len(channels), *shape)
                if logits is None or tuple(logits.shape) != expected or not torch.isfinite(logits).all():
                    raise ValueError(f'logits must be finite {expected}; adapter must undo any resize/padding')
                probabilities = (torch.sigmoid(logits) if args.segmentation_mode == 'multilabel' else torch.softmax(logits, 1)).cpu().numpy()
            for index, (key, path, im) in enumerate(batch):
                masks = {}
                for name in classes:
                    if key not in indexes[name]:
                        if not args.allow_missing_masks:
                            raise ValueError(f'Missing {name} mask for {path}')
                        if args.segmentation_mode == 'multiclass':
                            raise ValueError('Multiclass requires complete masks to construct exclusive labels')
                        masks_missing.setdefault(name, []).append(str(path))
                        continue
                    mask = read_plane(indexes[name][key], is_mask=True)
                    if mask.shape != shape:
                        raise ValueError(f'Mask shape {mask.shape} != image {shape}: {indexes[name][key]}')
                    masks[name] = mask > args.mask_threshold
                if args.segmentation_mode == 'multiclass' and np.any(np.stack(list(masks.values())).sum(0) > 1):
                    raise ValueError('Overlapping masks cannot be evaluated as multiclass; use multilabel or explicit disjoint labels')
                if need_features:
                    z = fmap[index]
                    features['global'].append(z.mean((-2, -1)).cpu().numpy())
                    keys['global'].append(key)
                    for name, mask in masks.items():
                        if mask.sum() < args.min_class_pixels:
                            continue
                        weights = torch.nn.functional.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None].to(device), size=z.shape[-2:], mode='area')[0, 0]
                        pooled = (z * weights).sum((-2, -1)) / weights.sum()
                        features[name].append(pooled.cpu().numpy())
                        keys[name].append(key)
                if 'frozen_seg' in active:
                    accumulator.add(key, probabilities[index], masks, args.segmentation_mode, channels, args.seg_threshold, groups[key], save_masks)
                    accumulator.rows[-1]['input_image'] = str(path)
                    accumulator.rows[-1]['target_sources'] = {name: str(indexes[name][key]) for name in masks}
                    if save_original is not None:
                        accumulator.rows[-1]['original'] = save_original(key, path)
        progress(f'task model: {min(offset+args.batch_size, len(items))}/{len(items)}')
    dimension = len(features['global'][0]) if need_features else 0
    return {'features': {name: np.asarray(rows, dtype=float).reshape(-1, dimension) for name, rows in features.items()} if need_features else {},
            'keys': keys, 'segmentation': accumulator.result(), 'shapes': shapes, 'missing_masks': masks_missing}


def evaluate_task(datasets, subsets, classes, args, active, progress, artifacts=None):
    if args.task_checkpoint is None:
        raise EncoderUnavailable('Task metrics need --task-checkpoint; see TASK_EXPERIMENT.md')
    if 'global' in classes or 'background' in classes:
        raise ValueError('Task metrics reserve class names global and background')
    if 'frozen_seg' in active and not args.model_classes:
        raise ValueError('Frozen segmentation requires explicit --model-classes in checkpoint channel order')
    if 'c2st' in active and not (args.groups_json or args.group_pattern):
        raise ValueError('C2ST requires --groups-json or --group-pattern to keep source slices together')
    model, torch, device, metadata = load_model(args)
    if args.require_real_only and metadata['training_provenance'] != 'real_only':
        raise ValueError(f'Task model provenance is {metadata["training_provenance"]}, not real_only')
    groups = source_groups(subsets, args)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(args.torch_threads)
    try:
        encoded = [encode_dataset(model, torch, device, subset, dataset[2], classes, args, active, group,
                                  lambda msg, label=label: progress(f'{label}: {msg}'),
                                  (lambda key, name, pred, target, label=label: artifacts.save_masks(label, key, name, pred, target)) if artifacts else None,
                                  (lambda key, path, label=label: artifacts.save_original(label, key, path)) if artifacts else None)
                   for label, dataset, subset, group in zip(('real', 'synthetic'), datasets, subsets, groups)]
    finally:
        torch.set_num_threads(old_threads)
    warnings = []
    if metadata['training_provenance'] != 'real_only':
        warnings.append('Task model training provenance is not real_only; these scores do not validate real-only -> synthetic transfer.')
    if not (args.groups_json or args.group_pattern):
        warnings.append('Task per-image groups default to image keys; do not bootstrap correlated crops as independent samples.')
    warnings.append('Task encoder scale/layer must stay fixed. EM-KID/FD are custom feature-space metrics, not Inception KID/FID.')
    scores = {}
    for metric in active:
        if metric == 'frozen_seg':
            real, synthetic = [e['segmentation'] for e in encoded]
            common = [c for c in classes if c in real['defined_classes'] and c in synthetic['defined_classes']]
            gap = {name: (mean_available([synthetic['classes'][c][name] for c in common]) - mean_available([real['classes'][c][name] for c in common])) if common else None
                   for name in ('dice', 'iou')}
            scores[metric] = {'real': real, 'synthetic': synthetic, 'synthetic_minus_real_macro': gap,
                              'gap_classes': common, 'boundary_tolerance_px': args.boundary_tolerance,
                              'direction': 'higher_accuracy_not_proof_of_realism',
                              'note': 'Pooled TP/FP/FN per class, then foreground macro. Both-empty class is null. '
                              'Per-image rows support source-group bootstrap. Entropy/Brier are diagnostics. '
                              'Positive transfer gap can mean easier synthetic scenes, not better realism.'}
            continue
        per_scope = {}
        for scope in ['global', *classes]:
            x, y = [e['features'][scope] for e in encoded]
            if metric == 'em_kid':
                score = kid(x, y, args.kid_subset_size, args.kid_subsets, args.seed)
            elif metric == 'em_fd':
                score = fid(x, y)
                if min(len(x), len(y)) <= x.shape[1]:
                    score['covariance_rank_deficient'] = True
            else:
                score = c2st(x, y, *[[g[k] for k in e['keys'][scope]] for g, e in zip(groups, encoded)], args.c2st_folds, args.seed)
            score.update(real_count=len(x), synthetic_count=len(y))
            per_scope[scope] = score
        value_key = {'em_kid': 'mean', 'em_fd': 'value', 'c2st': 'gap'}[metric]
        available = [c for c in classes if per_scope[c].get(value_key) is not None]
        scores[metric] = {'global': per_scope.pop('global'), 'classes': per_scope,
                          'macro_available': mean_available([per_scope[c].get(value_key) for c in classes]),
                          'macro_classes': available, 'requested_classes': classes,
                          'direction': 'lower',
                          'note': 'One pooled spatial feature per image/class with >= min_class_pixels. '
                          'Masks weight encoder features after inference; absent classes are omitted, not zeros. '
                          'Compare macro only with the same eligible classes.'}
    metadata.update(feature_pooling='global spatial mean and area-mask weighted mean per image/class',
                    group_source=str(args.groups_json) if args.groups_json else args.group_pattern or 'image_key_only',
                    input_contract='float32 Bx1xHxW in [0,1]; model adapter owns training preprocessing',
                    feature_keys={domain: e['keys'] for domain, e in zip(('real', 'synthetic'), encoded)},
                    groups=dict(zip(('real', 'synthetic'), groups)))
    return scores, metadata, encoded, warnings
