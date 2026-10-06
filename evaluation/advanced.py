"""Mask geometry and transparent, EM-oriented distribution descriptors."""
from itertools import combinations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from .data import read_intensity, read_plane
from .distances import rbf_mmd, sliced_wasserstein
from .metrics import distribution_distance


class PatchReservoir:
    """Uniform bounded reservoir of candidate patches across all input images."""
    def __init__(self, limit, dimensions, seed):
        self.rows = np.empty((limit, dimensions), dtype=np.float64)
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, row):
        self.seen += 1
        slot = self.seen - 1 if self.seen <= len(self.rows) else self.rng.integers(self.seen)
        if slot < len(self.rows):
            self.rows[slot] = row

    def values(self):
        return self.rows[:min(self.seen, len(self.rows))]


def descriptors(images, indexes, selected, args, progress):
    need_masks = bool({'spatial', 'joint_mmd'} & set(selected))
    reservoirs = {scale: PatchReservoir(args.swd_max_patches, args.patch_size ** 2, args.seed + scale)
                  for scale in (1, 2, 4)} if 'swd' in selected else {}
    spatial = {name: {'nearest_centroid_px': [], 'images_with_fewer_than_two_objects': 0,
                      'centroid_count': 0} for name in indexes}
    cross = {f'{a} -> {b}': [] for a in indexes for b in indexes if a != b}
    # Joint feature vectors per image preserve dependence between appearance and amount.
    joint, joint_keys, missing, shapes = [], [], {}, {}
    feature_names = ['image_mean', 'image_std']
    for name in indexes:
        feature_names.extend(f'{name}:{prop}' for prop in
                             ('coverage', 'foreground_mean', 'foreground_std', 'count_per_mpix', 'mean_area_px'))
    rng = np.random.default_rng(args.seed)
    for number, (key, path) in enumerate(images.items(), 1):
        image = read_intensity(path, args.intensity_max)
        shape = "x".join(map(str, image.shape))
        shapes[shape] = shapes.get(shape, 0) + 1
        if reservoirs:
            current = image
            for scale in (1, 2, 4):
                if scale != 1:
                    current = ndimage.gaussian_filter(current, sigma=1., mode='reflect')[::2, ::2]
                h, w = current.shape
                size = args.patch_size
                if min(h, w) < size:
                    continue
                for _ in range(args.patches_per_image):
                    y, x = rng.integers(h - size + 1), rng.integers(w - size + 1)
                    reservoirs[scale].add(current[y:y+size, x:x+size].ravel())
        centroids = {}
        vector = [float(image.mean()), float(image.std())]
        complete = True
        if need_masks:
            for name, index in indexes.items():
                if key not in index:
                    if not args.allow_missing_masks:
                        raise ValueError(f'Missing {name} mask for {path}; use --allow-missing-masks to skip')
                    missing.setdefault(name, []).append(str(path))
                    complete = False
                    continue
                mask = read_plane(index[key], is_mask=True)
                if mask.shape != image.shape:
                    raise ValueError(f'Mask shape {mask.shape} != image {image.shape}: {index[key]}')
                mask = mask > args.mask_threshold
                labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
                regions = regionprops(labels)
                if 'joint_mmd' in selected:
                    foreground = image[mask]
                    # Empty appearance is encoded as 0, together with explicit zero coverage/count.
                    vector.extend([float(mask.mean()), float(foreground.mean()) if foreground.size else 0.,
                                   float(foreground.std()) if foreground.size else 0., count * 1e6 / mask.size,
                                   float(mask.sum() / count) if count else 0.])
                if 'spatial' in selected:
                    points = np.array([region.centroid for region in regions], dtype=float).reshape(-1, 2)
                    centroids[name] = points
                    spatial[name]['centroid_count'] += len(points)
                    if len(points) >= 2:
                        distance = cKDTree(points).query(points, k=2)[0][:, 1]
                        spatial[name]['nearest_centroid_px'].extend(distance.tolist())
                    else:
                        spatial[name]['images_with_fewer_than_two_objects'] += 1
            if 'spatial' in selected:
                for a, b in combinations(centroids, 2):
                    if len(centroids[a]) and len(centroids[b]):
                        cross[f'{a} -> {b}'].extend(cKDTree(centroids[b]).query(centroids[a])[0].tolist())
                        cross[f'{b} -> {a}'].extend(cKDTree(centroids[a]).query(centroids[b])[0].tolist())
            if 'joint_mmd' in selected and complete:
                joint.append(vector)
                joint_keys.append(key)
        if number % 25 == 0 or number == len(images):
            progress(f'advanced: {number}/{len(images)} images processed')
    return {'patches': {scale: reservoir.values() for scale, reservoir in reservoirs.items()},
            'patch_candidates': {scale: reservoir.seen for scale, reservoir in reservoirs.items()},
            'spatial': spatial, 'cross': cross, 'missing_masks': missing,
            'joint': np.asarray(joint).reshape(-1, len(feature_names)), 'joint_keys': joint_keys,
            'feature_names': feature_names, 'shapes': shapes}


def compare_descriptors(a, b, selected, args):
    result = {}
    if 'swd' in selected:
        result['swd'] = {'variant': 'gray_gaussian_pyramid_raw_patches_v1', 'scales': {},
                         'note': 'Custom SWD descriptor; not the normalized Laplacian-pyramid ProGAN score.'}
        for scale in (1, 2, 4):
            score = sliced_wasserstein(a['patches'][scale], b['patches'][scale], args.swd_projections, args.seed)
            score['real_candidates'] = a['patch_candidates'][scale]
            score['synthetic_candidates'] = b['patch_candidates'][scale]
            result['swd']['scales'][str(scale)] = score
    if 'spatial' in selected:
        within = {}
        for name in a['spatial']:
            aa, bb = a['spatial'][name], b['spatial'][name]
            within[name] = distribution_distance(aa['nearest_centroid_px'], bb['nearest_centroid_px'])
            within[name]['real_centroid_count'] = aa['centroid_count']
            within[name]['synthetic_centroid_count'] = bb['centroid_count']
            within[name]['real_images_with_fewer_than_two_objects'] = aa['images_with_fewer_than_two_objects']
            within[name]['synthetic_images_with_fewer_than_two_objects'] = bb['images_with_fewer_than_two_objects']
        result['spatial'] = {'within_class_nearest_centroid_px': within,
                             'cross_class_nearest_centroid_px': {
                                 key: distribution_distance(a['cross'][key], b['cross'][key]) for key in a['cross']},
                             'note': 'Centroid distances in pixels, including border components; no edge correction. '
                                     'Not membrane-to-surface distances or a pair-correlation function.'}
    if 'joint_mmd' in selected:
        x, y = a['joint'], b['joint']
        if min(len(x), len(y)) < 2:
            result['joint_mmd'] = {'value': None, 'status': 'insufficient_samples',
                                    'minimum_per_dataset': 2, 'real_count': len(x), 'synthetic_count': len(y)}
        else:
            # Reference-only scaling: changing the synthetic set does not change the metric's ruler.
            mean, deviation = x.mean(axis=0), x.std(axis=0)
            constant = deviation < 1e-8
            deviation[constant] = 1.
            sigma = np.sqrt(x.shape[1])
            score = rbf_mmd((x - mean) / deviation, (y - mean) / deviation, sigma=sigma)
            score.update({'features': a['feature_names'], 'reference_mean': mean.tolist(),
                          'reference_scale': deviation.tolist(), 'constant_reference_features':
                          [name for name, is_constant in zip(a['feature_names'], constant) if is_constant],
                          'variant': 'image_and_mask_joint_descriptors_v1',
                          'note': 'Custom MMD, not FRD. Only images with all selected masks participate; '
                                  'reference-constant features use scale 1 in their original units.'})
            result['joint_mmd'] = score
    return result
