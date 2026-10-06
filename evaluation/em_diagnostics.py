"""Unpaired EM geometry, frequency and high-pass residual diagnostics."""
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, euler_number
from skimage.morphology import medial_axis

from .data import read_intensity, read_plane
from .metrics import distribution_distance, histogram_distance

DIAGNOSTICS = ('shape', 'psd', 'noise')


def shape_features(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    border = set(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))) - {0}
    out = {name: [] for name in ('area_px', 'perimeter_px', 'major_axis_px', 'minor_axis_px',
                                 'eccentricity', 'solidity', 'circularity', 'thickness_px')}
    out.update(coverage=[float(mask.mean())], count_per_mpix=[count * 1e6 / mask.size],
               euler_per_mpix=[euler_number(mask, connectivity=2) * 1e6 / mask.size])
    for region in regionprops(labels):
        if region.label in border:
            continue
        values = (region.area, region.perimeter, region.axis_major_length, region.axis_minor_length,
                  region.eccentricity, region.solidity)
        for name, value in zip(list(out)[:6], values):
            out[name].append(float(value))
        if region.perimeter > 0:
            out['circularity'].append(float(4 * np.pi * region.area / region.perimeter ** 2))
    # Remove clipped components before thickness measurement; zero-pad for a defined exterior.
    interior = mask & ~np.isin(labels, list(border))
    skeleton, distance = medial_axis(np.pad(interior, 1), return_distance=True, rng=0)
    out['thickness_px'] = (2 * distance[skeleton]).tolist()
    out['skeleton_pixels_per_mpix'] = [float(skeleton.sum() * 1e6 / mask.size)]
    return out, len(border)


def power_profiles(image, bins=32, angles=18):
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    power = np.abs(np.fft.fft2((image - image.mean()) * window)) ** 2 / max(float((window ** 2).sum()), 1e-12)
    fy, fx = np.fft.fftfreq(image.shape[0])[:, None], np.fft.fftfreq(image.shape[1])[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    valid = (radius > 0) & (radius <= .5)
    def profile(coordinate, n, limits):
        sums = np.histogram(coordinate[valid], bins=n, range=limits, weights=power[valid])[0]
        counts = np.histogram(coordinate[valid], bins=n, range=limits)[0]
        return np.divide(sums, counts, out=np.zeros(n), where=counts > 0)
    return (profile(radius, bins, (0, .5)),
            profile(np.mod(np.arctan2(fy, fx), np.pi), angles, (0, np.pi)))


def residual_features(image, sigma):
    residual = image - ndimage.gaussian_filter(image, sigma, mode='reflect')
    centered = residual - residual.mean()
    sd = float(centered.std())
    values = {'std': sd, 'mean': float(residual.mean()),
              'skew': float(np.mean((centered / sd) ** 3)) if sd > 1e-12 else None,
              'excess_kurtosis': float(np.mean((centered / sd) ** 4) - 3) if sd > 1e-12 else None}
    for axis in (0, 1):
        a, b = (residual[:-1], residual[1:]) if axis == 0 else (residual[:, :-1], residual[:, 1:])
        a, b = a - a.mean(), b - b.mean()
        denominator = float(np.sqrt((a*a).sum() * (b*b).sum()))
        values[f'autocorrelation_axis{axis}_lag1'] = float((a*b).sum() / denominator) if denominator > 1e-12 else None
    return values, np.histogram(residual, bins=128, range=(-1, 1))[0]


def extract_diagnostics(images, indexes, selected, args, progress):
    out = {'shape': {name: {} for name in indexes}, 'border': dict.fromkeys(indexes, 0),
           'radial': [], 'angular': [], 'noise': {}, 'residual_hist': np.zeros(128),
           'constant_residual_images': 0, 'shapes': {}, 'missing_masks': {}}
    for number, (key, path) in enumerate(images.items(), 1):
        image = read_intensity(path, args.intensity_max)
        shape = 'x'.join(map(str, image.shape))
        out['shapes'][shape] = out['shapes'].get(shape, 0) + 1
        if 'psd' in selected:
            radial, angular = power_profiles(image)
            out['radial'].append(radial)
            out['angular'].append(angular)
        if 'noise' in selected:
            values, hist = residual_features(image, args.noise_sigma)
            out['constant_residual_images'] += int(values['std'] <= 1e-12)
            out['residual_hist'] += hist
            for name, value in values.items():
                out['noise'].setdefault(name, [])
                if value is not None:
                    out['noise'][name].append(value)
        if 'shape' in selected:
            for name, index in indexes.items():
                if key not in index:
                    if not args.allow_missing_masks:
                        raise ValueError(f'Missing {name} mask for {path}')
                    out['missing_masks'].setdefault(name, []).append(str(path))
                    continue
                mask = read_plane(index[key], is_mask=True)
                if mask.shape != image.shape:
                    raise ValueError(f'Mask shape {mask.shape} != image {image.shape}: {index[key]}')
                values, border = shape_features(mask > args.mask_threshold)
                out['border'][name] += border
                for feature, samples in values.items():
                    out['shape'][name].setdefault(feature, []).extend(samples)
        if number % 25 == 0 or number == len(images):
            progress(f'EM diagnostics: {number}/{len(images)}')
    return out


def compare_diagnostics(a, b, selected, args):
    result = {}
    if 'shape' in selected:
        classes = {}
        for name in a['shape']:
            distances = {}
            for feature, x in a['shape'][name].items():
                d = distribution_distance(x, b['shape'][name].get(feature, []))
                if x:
                    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
                    scale = iqr if iqr > 1e-8 else (float(np.std(x)) if np.std(x) > 1e-8 else 1.)
                    d.update(reference_scale=scale, scale_fallback=iqr <= 1e-8,
                             normalized_wasserstein=d['wasserstein'] / scale if d['wasserstein'] is not None else None)
                else:
                    d.update(reference_scale=None, normalized_wasserstein=None)
                distances[feature] = d
            values = [d['normalized_wasserstein'] for d in distances.values() if d['normalized_wasserstein'] is not None]
            classes[name] = {'features': distances, 'mean_available_normalized_wasserstein': float(np.mean(values)) if values else None,
                             'available_features': len(values), 'total_features': len(distances),
                             'excluded_border_components': {'real': a['border'][name], 'synthetic': b['border'][name]}}
        means = [d['mean_available_normalized_wasserstein'] for d in classes.values()
                 if d['mean_available_normalized_wasserstein'] is not None]
        result['shape'] = {'classes': classes, 'macro_available': float(np.mean(means)) if means else None,
                           'direction': 'lower', 'variant': 'reference_scaled_geometry_v1',
                           'note': 'Equal feature/class weights are diagnostic, not calibrated utility. '
                           'Missing object distributions are null, not matches. Compare identical available features. '
                           'Scale: real IQR, fallback real std, fallback 1 original unit. '
                           '8-connected components; clipped objects excluded from shape/thickness. '
                           'Thickness = 2*EDT on medial axis, a raster approximation. Skeleton length is pixel count.'}
    if 'psd' in selected:
        x, y = np.log1p(a['radial']), np.log1p(b['radial'])
        mx, my = np.median(x, axis=0), np.median(y, axis=0)
        result['psd'] = {'radial_log_power_l1': float(np.mean(np.abs(mx-my))),
                         'radial_centered_log_power_l1': float(np.mean(np.abs((mx-mx.mean())-(my-my.mean())))),
                         'radial_real_median': mx.tolist(), 'radial_synthetic_median': my.tolist(),
                         'frequency_bin_edges_cycles_per_pixel': np.linspace(0, .5, 33).tolist(),
                         'angular': histogram_distance(np.sum(a['angular'], axis=0), np.sum(b['angular'], axis=0)),
                         'angle_bin_edges_radians': np.linspace(0, np.pi, 19).tolist(), 'direction': 'lower',
                         'note': 'Hann window, DC excluded, Nyquist disk, FFT power / sum(window**2). '
                         'Median log1p radial power; angular mean-power bins retain directional artifacts. '
                         'Not a calibrated physical noise PSD; equal nm/pixel required.'}
    if 'noise' in selected:
        result['noise'] = {'features': {name: distribution_distance(x, b['noise'][name]) for name, x in a['noise'].items()},
                           'residual_histogram': histogram_distance(a['residual_hist'], b['residual_hist']),
                           'constant_residual_images': {'real': a['constant_residual_images'], 'synthetic': b['constant_residual_images']},
                           'sigma_px': args.noise_sigma, 'direction': 'lower',
                           'note': 'Image minus Gaussian blur; contains edges/texture as well as acquisition noise. '
                           'Skew, kurtosis and correlation are undefined for constant residuals and omitted with counts.'}
    return result
