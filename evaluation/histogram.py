"""Equal-image-weight intensity distributions and headless comparison plots."""
import numpy as np

from .data import read_intensity, read_plane
from .artifacts import safe_relative
from .metrics import histogram, histogram_distance


def summarize(images, maximum, masks=None, threshold=0, allow_missing=False):
    total = np.zeros(256, dtype=float)
    shapes = {}
    present = empty = missing = pixels = 0
    for key, path in images.items():
        values = read_intensity(path, maximum)
        if masks is not None:
            if key not in masks:
                if not allow_missing:
                    raise ValueError(f'Missing mask for {path}')
                missing += 1
                continue
            mask = read_plane(masks[key], is_mask=True) > threshold
            if mask.shape != values.shape:
                raise ValueError(f'Mask/image shape mismatch: {masks[key]} and {path}')
            values = values[mask]
            if not values.size:
                empty += 1
                continue
        counts = histogram(values)
        total += counts / counts.sum()
        present += 1
        pixels += values.size
        shape = 'masked' if masks is not None else 'x'.join(map(str, values.shape))
        shapes[shape] = shapes.get(shape, 0) + 1
    if not images:
        raise ValueError('Histogram requires at least one image')
    return {'image_count': present, 'selected_images': len(images), 'empty_mask_images': empty,
            'missing_mask_images': missing, 'foreground_pixels': int(pixels),
            'mean_probability': (total / present).tolist() if present else None, 'shapes': shapes}


def compare_histograms(subsets, maximum, artifacts=None, *, masks=None, threshold=0,
                       allow_missing=False, class_name=None):
    real, synthetic = [summarize(images, maximum, domain_masks, threshold, allow_missing)
                       for images, domain_masks in zip(subsets, masks if masks is not None else (None, None))]
    valid = real['image_count'] > 0 and synthetic['image_count'] > 0
    p, q = [np.asarray(item['mean_probability']) for item in (real, synthetic)]
    result = {'status': 'ok' if valid else 'undefined_empty_class', 'class_name': class_name, 'aggregation': 'mean of individually normalized image histograms; equal image weights',
              'bins': 256, 'bin_edges': np.linspace(0, maximum, 257).tolist(),
              'intensity_max': maximum, 'real': real, 'synthetic': synthetic,
              'distances': histogram_distance(p, q, ordered=True) if valid else None,
              'overlap': float(np.minimum(p, q).sum()) if valid else None,
              'interpretation': 'Overlap: higher is closer (1 identical). Distances: lower is closer. '
                                'Wasserstein uses intensity normalized to [0,1]. No spatial similarity is measured.',
              'plots': {}}
    if artifacts is not None:
        result['plots'] = save_plots(result, artifacts)
    return result


def save_plots(result, artifacts):
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as error:
        raise ValueError('histogram plots require matplotlib; install evaluation/requirements-lite.txt') from error
    figure = Figure(figsize=(12, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    edges = np.asarray(result['bin_edges'])
    for domain, color in (('real', '#2563eb'), ('synthetic', '#ea580c')):
        stats = result[domain]
        if stats['mean_probability'] is None:
            axes[0].plot([], [], color=color, label=f'{domain}: no foreground')
            axes[1].plot([], [], color=color, label=f'{domain}: no foreground')
            continue
        probabilities = np.asarray(stats['mean_probability'])
        label = f"{domain} (n={stats['image_count']})"
        axes[0].stairs(probabilities, edges, color=color, label=label, linewidth=1.5)
        axes[1].plot(edges, np.r_[0., probabilities.cumsum()], color=color, label=label)
    for axis, title, ylabel in zip(axes, ('Mean intensity histogram', 'Cumulative distribution'),
                                  ('Mean pixel fraction per bin', 'Cumulative probability')):
        axis.set(title=title, xlabel='Pixel intensity', ylabel=ylabel, xlim=(0, result['intensity_max']))
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
        axis.legend()
    overlap = f"{result['overlap']:.4f}" if result['overlap'] is not None else 'undefined'
    title = result.get('class_name') or 'All pixels'
    figure.suptitle(f"{title} | Equal weight per nonempty image | overlap = {overlap}")
    directory = artifacts.directory / 'plots'
    if result.get('class_name') is not None:
        directory = directory / 'classes' / safe_relative(result['class_name'])
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in ('png', 'svg'):
        path = directory / f'histogram.{extension}'
        figure.savefig(path, dpi=180)
        paths[extension] = str(path.relative_to(artifacts.root))
    return paths
