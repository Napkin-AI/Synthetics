#!/usr/bin/env python3
"""Compare independent real/synthetic EM datasets; default: FID and KID."""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'evaluation'

from .advanced import compare_descriptors, descriptors
from .data import canonical_class, discover
from .distances import fid, kid, prdc, rbf_mmd, vendi
from .embeddings import Encoder, EncoderUnavailable, package_version
from .metrics import MASK_METRICS, METRICS as CLASSICAL, compare, extract
from .em_diagnostics import DIAGNOSTICS, extract_diagnostics, compare_diagnostics
from .task_metrics import TASK_METRICS, evaluate_task
from .artifacts import RunArtifacts
from .histogram import compare_histograms

ADVANCED = ('swd', 'spatial', 'joint_mmd')
NEURAL = ('fid', 'kid', 'prdc', 'vendi', 'cmmd')
MODES = {'basic': ('fid', 'kid'),
         'medium': ('fid', 'kid') + CLASSICAL + DIAGNOSTICS,
         'full': ('fid', 'kid') + CLASSICAL + DIAGNOSTICS + ('prdc', 'cmmd', 'vendi') + ADVANCED}
METRICS = MODES['full'] + TASK_METRICS + ('histogram',)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('real', type=Path, nargs='?', help='Real root containing original/ and class folders')
    p.add_argument('synthetic', type=Path, nargs='?', help='Synthetic root containing original/ and class folders')
    p.add_argument('--mode', choices=MODES, default='basic', help='Nested metric preset')
    p.add_argument('--metrics', nargs='+', choices=('all',) + METRICS, help='Replace the preset with an explicit list')
    p.add_argument('--exclude-metrics', nargs='+', choices=METRICS, default=[], help='Remove metrics from selected preset/list')
    p.add_argument('--list-metrics', action='store_true', help='Print presets and exit; no datasets or weights required')
    p.add_argument('--classes', nargs='+', help='Class folder names; default: common matched class folders')
    p.add_argument('--max-images', type=int, help='Seeded sample size per dataset; omitted means all images')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--intensity-max', type=float, default=255, help='Shared scale: 255, 65535, or 1; no per-image normalization')
    p.add_argument('--mask-threshold', type=float, default=0, help='Foreground = raw mask value > threshold')
    p.add_argument('--allow-missing-masks', action='store_true', help='Skip absent masks and report them')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N for neural encoders')
    p.add_argument('--batch-size', type=int, default=16, help='Images per inference batch; does not limit dataset size or batch count')
    p.add_argument('--torch-threads', type=int, default=4, help='CPU threads for neural encoders')
    p.add_argument('--cache-dir', type=Path, default=Path.home() / '.cache' / 'synthetics-evaluation' / 'v2')
    p.add_argument('--no-cache', action='store_true', help='Do not read/write feature cache')
    p.add_argument('--offline', action='store_true', help='Never download model weights; use existing models/features')
    p.add_argument('--skip-unavailable', action='store_true', help='Mark missing encoders as unavailable and continue')
    p.add_argument('--kid-subset-size', type=int, default=1000)
    p.add_argument('--kid-subsets', type=int, default=50)
    p.add_argument('--prdc-k', type=int, default=5)
    p.add_argument('--max-pairwise-samples', type=int, default=2000,
                   help='Seeded cap for PRDC, CMMD, Vendi, joint MMD; FID/KID still use all selected images')
    p.add_argument('--patch-size', type=int, default=7, help='SWD square patch side')
    p.add_argument('--patches-per-image', type=int, default=64, help='SWD candidate patches per scale per image')
    p.add_argument('--swd-max-patches', type=int, default=2048, help='Reservoir size per scale per dataset')
    p.add_argument('--swd-projections', type=int, default=128)
    p.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent/'results',
                   help='Root for report/<run>.json and segmentation/<run>/')
    p.add_argument('--run-info', help='Human-readable suffix for the date_config_info run name')
    p.add_argument('--output', type=Path, help='Optional extra copy of the JSON, in addition to results/report')
    p.add_argument('--quiet', action='store_true', help='Suppress evaluation progress on stderr')
    p.add_argument('--noise-sigma', type=float, default=1., help='Gaussian residual sigma in pixels')
    p.add_argument('--task-checkpoint', type=Path, help='Frozen segmentor/encoder checkpoint')
    p.add_argument('--task-adapter', help='module:function or /path/adapter.py:function; omitted: TorchScript')
    p.add_argument('--task-config', type=Path, help='JSON passed to adapter factory; include training provenance')
    p.add_argument('--require-real-only', action='store_true', help='Reject task models whose recorded training provenance is not real_only')
    p.add_argument('--model-classes', nargs='+', help='Exact model channel order; independent of --classes')
    p.add_argument('--segmentation-mode', choices=('multilabel', 'multiclass'), default='multilabel')
    p.add_argument('--seg-threshold', type=float, default=.5, help='Fixed multilabel probability threshold')
    p.add_argument('--boundary-tolerance', type=float, default=2., help='Boundary F1 tolerance in pixels')
    p.add_argument('--min-class-pixels', type=int, default=16, help='Minimum mask area for a class feature vector')
    p.add_argument('--groups-json', type=Path, help='Domain -> image key -> source slice/volume ID; required for C2ST unless regex supplied')
    p.add_argument('--group-pattern', help='Regex with one capture group extracting source ID from image key')
    p.add_argument('--c2st-folds', type=int, default=3, help='Held-out group folds per domain')
    return p


def sample(images, maximum, seed):
    if maximum is None or maximum >= len(images):
        return images
    keys = sorted(images)
    indexes = sorted(np.random.default_rng(seed).choice(len(keys), maximum, replace=False))
    return {keys[i]: images[keys[i]] for i in indexes}


def validate(args):
    if args.real is None or args.synthetic is None:
        raise ValueError('Both real and synthetic dataset paths are required')
    for name in ('intensity_max', 'mask_threshold'):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0 or (name == 'intensity_max' and value == 0):
            raise ValueError(f'Invalid --{name.replace("_", "-")}')
    for name in ('max_images', 'batch_size', 'torch_threads', 'kid_subsets', 'prdc_k',
                 'patch_size', 'patches_per_image', 'swd_max_patches', 'swd_projections'):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be >= 1')
    if args.kid_subset_size < 2 or args.max_pairwise_samples < 2:
        raise ValueError('--kid-subset-size and --max-pairwise-samples must be >= 2')
    if args.seed < 0:
        raise ValueError('--seed must be nonnegative')
    if not re.fullmatch(r'auto|cpu|cuda(?::\d+)?', args.device):
        raise ValueError('--device must be auto, cpu, cuda, or cuda:N')
    if not np.isfinite(args.noise_sigma) or args.noise_sigma <= 0:
        raise ValueError('--noise-sigma must be finite and positive')
    if not np.isfinite(args.boundary_tolerance) or args.boundary_tolerance < 0:
        raise ValueError('--boundary-tolerance must be finite and nonnegative')
    if not 0 < args.seg_threshold < 1:
        raise ValueError('--seg-threshold must be between 0 and 1')
    if args.c2st_folds < 2 or args.min_class_pixels < 1:
        raise ValueError('--c2st-folds >= 2 and --min-class-pixels >= 1 required')


def evaluate(args, artifacts=None):
    start = time.perf_counter()
    validate(args)
    requested = METRICS if args.metrics and 'all' in args.metrics else (args.metrics or MODES[args.mode])
    # A supplied segmentor is evaluated unless an explicit metric list/exclusion says otherwise.
    if args.task_checkpoint and args.metrics is None:
        requested = (*requested, 'frozen_seg')
    selected = [name for name in dict.fromkeys(requested) if name not in args.exclude_metrics]
    if not selected:
        raise ValueError('No metrics selected')
    need_masks = bool(set(selected) & (MASK_METRICS | {'spatial', 'joint_mmd', 'shape'} | set(TASK_METRICS)))
    inspect_masks = need_masks or 'histogram' in selected
    datasets = [discover(args.real, with_masks=inspect_masks), discover(args.synthetic, with_masks=inspect_masks)]
    warnings = []
    real_classes, synth_classes = datasets[0][2], datasets[1][2]
    classes = sorted(real_classes.keys() & synth_classes.keys()) if inspect_masks else []
    if args.classes:
        if not inspect_masks:
            raise ValueError('--classes requires a mask-based metric')
        classes = list(dict.fromkeys(canonical_class(name) for name in args.classes))
        missing = set(classes) - (real_classes.keys() & synth_classes.keys())
        if missing:
            raise ValueError(f'Requested classes missing or without matching masks: {sorted(missing)}')
    if need_masks and not classes:
        raise ValueError('No common classes with matching masks; check names or select image-only metrics')
    if inspect_masks and real_classes.keys() != synth_classes.keys():
        warnings.append(f'Only selected common classes are compared. Real-only: '
                        f'{sorted(real_classes.keys() - synth_classes.keys())}; synthetic-only: '
                        f'{sorted(synth_classes.keys() - real_classes.keys())}')
    subsets = [sample(dataset[1], args.max_images, args.seed) for dataset in datasets]
    metadata = []
    for label, dataset, subset in zip(('real', 'synthetic'), datasets, subsets):
        if dataset[3]:
            warnings.append(f'{label}: ignored folders without matching masks: {dataset[3]}')
        metadata.append({'root': str(dataset[0]), 'available_images': len(dataset[1]), 'used_images': len(subset),
                         'files': [str(path) for path in subset.values()], 'shapes': {}, 'missing_masks': {}})
        if len(subset) < 10:
            warnings.append(f'{label}: only {len(subset)} images; estimates may be unstable')
        if inspect_masks and not args.allow_missing_masks:
            for name in classes:
                missing = subset.keys() - dataset[2][name].keys()
                if missing:
                    raise ValueError(f'Missing {name} mask for {subset[sorted(missing)[0]]}; '
                                     'use --allow-missing-masks to skip')
    def progress(message):
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)
    scores, encoders = {}, {}
    if 'histogram' in selected:
        progress('Computing equal-image-weight intensity histograms')
        scores['histogram'] = compare_histograms(subsets, args.intensity_max, artifacts)
        for label, meta in zip(('real', 'synthetic'), metadata):
            meta['shapes'] = scores['histogram'][label]['shapes']
        scores['histogram']['classes'] = {}
        if not classes:
            warnings.append('histogram: no common annotated classes; only global distributions plotted')
        for name in classes:
            progress(f'Computing intensity histograms inside {name} masks')
            comparison = compare_histograms(
                subsets, args.intensity_max, artifacts,
                masks=[dataset[2][name] for dataset in datasets], threshold=args.mask_threshold,
                allow_missing=args.allow_missing_masks, class_name=name)
            scores['histogram']['classes'][name] = comparison
            for label, meta in zip(('real', 'synthetic'), metadata):
                count = comparison[label]['missing_mask_images']
                if count:
                    meta['missing_masks'][name] = count
    task_selected = [name for name in selected if name in TASK_METRICS]
    if task_selected:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                task_scores, task_metadata, task_encoded, task_warnings = evaluate_task(
                    datasets, subsets, classes, args, task_selected, progress, artifacts)
            scores.update(task_scores)
            encoders['task'] = task_metadata
            warnings.extend(task_warnings)
            for meta, features in zip(metadata, task_encoded):
                meta['shapes'] = features['shapes']
                meta['missing_masks'].update(features['missing_masks'])
        except EncoderUnavailable as error:
            if not args.skip_unavailable:
                raise
            for name in task_selected:
                scores[name] = {'value': None, 'status': 'unavailable', 'reason': str(error)}
            warnings.append(f'Task model unavailable: {error}')
    for encoder_name, metric_names in (('inception', ('fid', 'kid', 'prdc', 'vendi')), ('clip', ('cmmd',))):
        active = [name for name in metric_names if name in selected]
        if not active:
            continue
        encoder = None
        try:
            with contextlib.redirect_stdout(sys.stderr):
                encoder = Encoder(encoder_name, args)
                old_threads = encoder.torch.get_num_threads()
                encoder.torch.set_num_threads(args.torch_threads)
                try:
                    encoded = [encoder.features(subset, lambda message, label=label: progress(f'{label}: {message}'))
                               for label, subset in zip(('real', 'synthetic'), subsets)]
                finally:
                    encoder.torch.set_num_threads(old_threads)
            encoders[encoder_name] = {'specification': encoder.specification,
                                      'cache': {label: value[2] for label, value in zip(('real', 'synthetic'), encoded)}}
            for meta, (_, shapes, _) in zip(metadata, encoded):
                unique, counts = np.unique(shapes, axis=0, return_counts=True)
                meta['shapes'] = {'x'.join(map(str, shape)): int(count) for shape, count in zip(unique, counts)}
            x, y = encoded[0][0], encoded[1][0]
            if 'fid' in active:
                scores['fid'] = fid(x, y)
                if min(len(x), len(y)) <= x.shape[1]:
                    warnings.append('FID: sample covariance is rank deficient; small-set scores are exploratory')
            if 'kid' in active:
                scores['kid'] = kid(x, y, args.kid_subset_size, args.kid_subsets, args.seed)
            # Bounded quadratic metrics use the same reproducible subset of each dataset.
            bounded = [sample(dict(enumerate(z)), args.max_pairwise_samples, args.seed) for z in (x, y)]
            xx, yy = [np.asarray(list(z.values())) for z in bounded]
            for name in active:
                if name == 'prdc':
                    scores[name] = prdc(xx, yy, args.prdc_k)
                elif name == 'vendi':
                    scores[name] = {'real': vendi(xx), 'synthetic': vendi(yy), 'direction': 'compare_diversity_not_quality'}
                elif name == 'cmmd':
                    scores[name] = rbf_mmd(xx, yy, sigma=10., scale=1000.)
                    scores[name]['variant'] = 'clip_vit_l14_336_torch_bicubic_v1'
                    scores[name]['note'] = 'Released CMMD V-statistic; PyTorch preprocessing, not bit-identical to JAX reference.'
                if name in ('prdc', 'vendi', 'cmmd'):
                    scores[name]['sample_indices'] = dict(zip(('real', 'synthetic'), [list(z) for z in bounded]))
        except EncoderUnavailable as error:
            if not args.skip_unavailable:
                raise EncoderUnavailable(f'{error}\nUse --skip-unavailable for a partial report, or --metrics for CPU-only metrics.') from error
            for name in active:
                scores[name] = {'value': None, 'status': 'unavailable', 'reason': str(error)}
            warnings.append(f'{encoder_name} unavailable; requested metrics are explicitly marked as missing')
        finally:
            if encoder is not None:
                del encoder
                gc.collect()
    classical_selected = [name for name in selected if name in CLASSICAL]
    if classical_selected:
        features = []
        for label, dataset, subset, meta in zip(('real', 'synthetic'), datasets, subsets, metadata):
            feature = extract(subset, {name: dataset[2][name] for name in classes}, classical_selected, args,
                              lambda message, label=label: progress(f'{label}: {message}'))
            meta['shapes'], meta['missing_masks'] = feature['shapes'], feature['missing_masks']
            features.append(feature)
        scores.update(compare(*features, classical_selected))
    advanced_selected = [name for name in selected if name in ADVANCED]
    if advanced_selected:
        features = []
        for label, dataset, subset, meta in zip(('real', 'synthetic'), datasets, subsets, metadata):
            feature = descriptors(subset, {name: dataset[2][name] for name in classes}, advanced_selected, args,
                                  lambda message, label=label: progress(f'{label}: {message}'))
            meta['missing_masks'].update(feature['missing_masks'])
            meta['shapes'] = feature['shapes']
            if 'joint_mmd' in selected:
                bounded = sample(dict(zip(feature['joint_keys'], feature['joint'])), args.max_pairwise_samples, args.seed)
                feature['joint'] = np.asarray(list(bounded.values())).reshape(-1, len(feature['feature_names']))
                meta['joint_mmd_image_keys'] = list(bounded)
            features.append(feature)
        scores.update(compare_descriptors(*features, advanced_selected, args))
    diagnostic_selected = [name for name in selected if name in DIAGNOSTICS]
    if diagnostic_selected:
        features = []
        for label, dataset, subset, meta in zip(('real', 'synthetic'), datasets, subsets, metadata):
            feature = extract_diagnostics(subset, {name: dataset[2][name] for name in classes}, diagnostic_selected, args,
                                          lambda msg, label=label: progress(f'{label}: {msg}'))
            meta['shapes'] = feature['shapes']
            meta['missing_masks'].update(feature['missing_masks'])
            features.append(feature)
        scores.update(compare_diagnostics(*features, diagnostic_selected, args))
    for label, meta in zip(('real', 'synthetic'), metadata):
        if meta['missing_masks']:
            warnings.append(f'{label}: missing masks skipped; class estimates may be biased')
    if metadata[0]['shapes'].keys() != metadata[1]['shapes'].keys():
        warnings.append('Image sizes differ; physical field of view, resizing for encoders, and clipping can affect metrics')
    warnings.append('Ensure comparable nm/pixel and annotation rules. Adjacent slices and patches are correlated observations.')
    if any(len(subset) > args.max_pairwise_samples for subset in subsets) and set(selected) & {'prdc', 'cmmd', 'vendi', 'joint_mmd'}:
        warnings.append('Quadratic metrics were capped by --max-pairwise-samples; see recorded sample indices/keys')
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
              if name not in ('real', 'synthetic', 'output', 'quiet', 'list_metrics')}
    config.update({'effective_mode': 'custom' if args.metrics or args.exclude_metrics else args.mode,
                   'metrics': selected, 'classes': classes})
    return {'schema_version': 2, 'status': 'partial' if any(v.get('status') == 'unavailable' for v in scores.values()) else 'complete',
            'metrics': {name: scores[name] for name in selected}, 'datasets': dict(zip(('real', 'synthetic'), metadata)),
            'config': config, 'encoders': encoders,
            'interpretation': 'Distances: lower is closer. PRDC: higher fidelity/coverage. Vendi measures diversity, not realism. '
                              'Null/undefined is not a perfect match. No universal quality threshold or aggregate score.',
            'warnings': warnings, 'elapsed_seconds': time.perf_counter() - start,
            'versions': {name: package_version(name) for name in ('numpy', 'scipy', 'Pillow', 'scikit-image',
                                                                'torch', 'torchvision', 'pytorch-fid', 'transformers')}}


def main(argv=None):
    p = parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(arguments)
    if args.list_metrics:
        print(json.dumps({**MODES, 'optional_task_metrics': TASK_METRICS, 'optional_visual_metrics': ['histogram']}, indent=2))
        return
    artifacts = None
    try:
        validate(args)
        artifacts = RunArtifacts(args, [sys.executable, str(Path(__file__).resolve()), *arguments])
        result = evaluate(args, artifacts)
        artifacts.finish(result)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + '\n', encoding='utf-8')
        print(payload)
    except BaseException as error:
        if artifacts is not None and artifacts.state['status'] == 'running':
            artifacts.fail(error)
        if isinstance(error, (ValueError, OSError, EncoderUnavailable)):
            p.exit(2, f'error: {error}\n')
        raise


if __name__ == '__main__':
    main()
