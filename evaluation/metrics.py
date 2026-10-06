"""Streaming descriptors and distances for unpaired EM datasets."""
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops

from .data import read_intensity, read_plane

METRICS = ('intensity', 'class_intensity', 'lbp', 'glcm', 'spectrum', 'coverage', 'morphology')
MASK_METRICS = {'class_intensity', 'coverage', 'morphology'}
GLCM_PROPERTIES = ('contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation')


def histogram(values, bins=256):
    return np.histogram(values, bins=bins, range=(0, 1))[0].astype(float)


def histogram_distance(a, b, ordered=False):
    if a.sum() == 0 or b.sum() == 0:
        return {'status': 'undefined_empty_distribution', 'real_count': int(a.sum()),
                'synthetic_count': int(b.sum())}
    p, q = a / a.sum(), b / b.sum()
    result = {'js_distance': float(jensenshannon(p, q, base=2)),
              'hellinger': float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)),
              'total_variation': float(np.abs(p - q).sum() / 2)}
    if ordered:
        centers = (np.arange(len(p)) + 0.5) / len(p)
        result['wasserstein'] = float(wasserstein_distance(centers, centers, p, q))
        result['cdf_max_distance'] = float(np.abs(np.cumsum(p) - np.cumsum(q)).max())
    return result


def summary(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {'count': 0, 'mean': None, 'std': None, 'median': None}
    return {'count': len(x), 'mean': float(x.mean()), 'std': float(x.std()),
            'median': float(np.median(x))}


def distribution_distance(a, b):
    result = {'real': summary(a), 'synthetic': summary(b)}
    result['wasserstein'] = float(wasserstein_distance(a, b)) if len(a) and len(b) else None
    if not len(a) or not len(b):
        result['status'] = 'undefined_empty_distribution'
    return result


def spectrum(image, bins=64):
    centered = image - image.mean()
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    power = np.abs(np.fft.fft2(centered * window)) ** 2
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    # Isotropic disk up to Nyquist; ignore corner frequencies and DC.
    valid = (radius > 0) & (radius <= .5)
    totals = np.histogram(radius[valid], bins=bins, range=(0, .5), weights=power[valid])[0]
    counts = np.histogram(radius[valid], bins=bins, range=(0, .5))[0]
    radial = np.divide(totals, counts, out=np.zeros(bins), where=counts > 0)
    total = radial.sum()
    return radial / total if total > 0 else radial


def extract(images, mask_indexes, selected, args, progress):
    out = {'intensity': np.zeros(256), 'lbp': np.zeros(10), 'spectrum': np.zeros(64),
           'glcm': {}, 'classes': {}, 'files': [], 'shapes': {}, 'missing_masks': {},
           'spectrum_nonzero_images': 0}
    need_masks = bool(set(selected) & MASK_METRICS)
    for name in mask_indexes:
        out['classes'][name] = {'intensity': np.zeros(256), 'coverage': [], 'density_per_mpix': [],
                               'area_px': [], 'eccentricity': [], 'solidity': [],
                               'equivalent_diameter_px': [], 'border_components': 0}
    for number, (key, path) in enumerate(images.items(), 1):
        image = read_intensity(path, args.intensity_max)
        out['files'].append(str(path))
        shape = 'x'.join(map(str, image.shape))
        out['shapes'][shape] = out['shapes'].get(shape, 0) + 1
        if 'intensity' in selected:
            out['intensity'] += histogram(image)
        if 'lbp' in selected or 'glcm' in selected:
            u8 = np.rint(image * 255).astype(np.uint8)
        if 'lbp' in selected:
            # Exclude one-pixel boundary affected by padding.
            lbp = local_binary_pattern(u8, 8, 1, method='uniform')[1:-1, 1:-1]
            out['lbp'] += np.bincount(lbp.astype(int).ravel(), minlength=10)
        if 'glcm' in selected:
            quantized = (u8 // 8).astype(np.uint8)
            for distance in (1, 2, 4):
                if min(image.shape) <= distance:
                    continue
                matrix = graycomatrix(quantized, [distance], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                      levels=32, symmetric=True, normed=True)
                for prop in GLCM_PROPERTIES:
                    values = graycoprops(matrix, prop)[0]
                    # Keep orientations separate so anisotropy is not averaged away.
                    for angle, value in enumerate(values):
                        feature = f'{prop}_d{distance}_angle{angle * 45}'
                        out['glcm'].setdefault(feature, []).append(float(value))
        if 'spectrum' in selected:
            radial = spectrum(image)
            out['spectrum'] += radial
            out['spectrum_nonzero_images'] += int(radial.sum() > 0)
        if need_masks:
            for name, index in mask_indexes.items():
                if key not in index:
                    if not args.allow_missing_masks:
                        raise ValueError(f'Missing {name} mask for {path}; use --allow-missing-masks to skip')
                    out['missing_masks'].setdefault(name, []).append(str(path))
                    continue
                mask = read_plane(index[key], is_mask=True)
                if mask.shape != image.shape:
                    raise ValueError(f'Mask shape {mask.shape} != image {image.shape}: {index[key]}')
                mask = mask > args.mask_threshold
                values = out['classes'][name]
                if 'class_intensity' in selected:
                    values['intensity'] += histogram(image[mask])
                if 'coverage' in selected:
                    values['coverage'].append(float(mask.mean()))
                if 'morphology' in selected:
                    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
                    # Density includes border objects; shape statistics exclude clipped objects.
                    values['density_per_mpix'].append(count * 1e6 / mask.size)
                    border = set(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))) - {0}
                    values['border_components'] += len(border)
                    for region in regionprops(labels):
                        if region.label in border:
                            continue
                        values['area_px'].append(float(region.area))
                        values['eccentricity'].append(float(region.eccentricity))
                        values['solidity'].append(float(region.solidity))
                        values['equivalent_diameter_px'].append(float(region.equivalent_diameter_area))
        if number % 25 == 0 or number == len(images):
            progress(f'{number}/{len(images)} images processed')
    return out


def compare(real, synthetic, selected):
    result = {}
    for metric in selected:
        if metric in ('intensity', 'lbp', 'spectrum'):
            result[metric] = histogram_distance(real[metric], synthetic[metric], ordered=metric == 'intensity')
            if metric == 'spectrum':
                result[metric]['real_nonzero_images'] = real['spectrum_nonzero_images']
                result[metric]['synthetic_nonzero_images'] = synthetic['spectrum_nonzero_images']
        elif metric == 'glcm':
            keys = sorted(real['glcm'].keys() | synthetic['glcm'].keys())
            result[metric] = {key: distribution_distance(real['glcm'].get(key, []), synthetic['glcm'].get(key, []))
                              for key in keys}
        else:
            result[metric] = {}
            for name in real['classes']:
                a, b = real['classes'][name], synthetic['classes'][name]
                if metric == 'class_intensity':
                    result[metric][name] = histogram_distance(a['intensity'], b['intensity'], ordered=True)
                elif metric == 'coverage':
                    result[metric][name] = distribution_distance(a['coverage'], b['coverage'])
                else:
                    features = ('density_per_mpix', 'area_px', 'equivalent_diameter_px', 'eccentricity', 'solidity')
                    result[metric][name] = {feature: distribution_distance(a[feature], b[feature]) for feature in features}
                    result[metric][name]['excluded_border_components'] = {
                        'real': a['border_components'], 'synthetic': b['border_components']}
    return result
