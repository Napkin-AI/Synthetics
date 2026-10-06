"""Strict 2D dataset loading; pairing happens only within each dataset."""
from pathlib import Path
import os
import re

import numpy as np
from PIL import Image

EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}


def canonical_class(name):
    return re.sub(r'[\s_]+', '_', name.strip()).casefold()


def image_key(relative):
    stem = relative.stem
    # Known Lucchi naming convention; never match arbitrary trailing numbers.
    match = re.fullmatch(r'(?:testing|predict_test)(\d+)', stem)
    if match:
        stem = 'lucchi_' + str(int(match.group(1)))
    return str(relative.parent / stem)


def index_images(directory):
    result = {}
    # os.walk uses directory entries instead of a stat call per file on /mnt/d.
    for parent, directories, filenames in os.walk(directory):
        directories.sort()
        for filename in sorted(filenames):
            path = Path(parent) / filename
            if path.suffix.lower() not in EXTENSIONS:
                continue
            key = image_key(path.relative_to(directory))
            if key in result:
                raise ValueError(f'Ambiguous image key {key}: {result[key]} and {path}')
            result[key] = path
    return result


def discover(root, *, with_masks=True):
    root = Path(root).resolve()
    images = index_images(root / 'original')
    if not images:
        raise ValueError(f'No images in {root / "original"}')
    classes, ignored = {}, []
    if not with_masks:
        return root, images, classes, ignored
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name == 'original' or directory.name.startswith('.'):
            continue
        masks = index_images(directory)
        if not (masks.keys() & images.keys()):
            ignored.append(directory.name)
            continue
        name = canonical_class(directory.name)
        if name in classes:
            raise ValueError(f'Duplicate normalized class {name} in {root}')
        classes[name] = masks
    return root, images, classes, ignored


def read_plane(path, *, is_mask=False):
    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError(f'{path}: multipage images are unsupported; export separate 2D slices')
        if image.mode == 'P':
            image = image.convert('RGBA' if 'transparency' in image.info else 'RGB')
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] in (2, 3, 4):
        channels = array.shape[2]
        if channels in (3, 4) and not (np.array_equal(array[..., 0], array[..., 1]) and
                                       np.array_equal(array[..., 0], array[..., 2])):
            raise ValueError(f'{path}: expected grayscale or equal RGB channels')
        if channels in (2, 4):
            alpha = array[..., -1]
            if not is_mask and not np.all(alpha == 255):
                raise ValueError(f'{path}: transparent images are unsupported')
            # Composite grayscale masks onto black background, preserving alpha.
            array = array[..., 0].astype(float) * (alpha.astype(float) / 255)
        else:
            array = array[..., 0]
    if array.ndim != 2 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError(f'{path}: expected finite 2D image with dimensions >= 2')
    return array


def read_intensity(path, maximum):
    array = read_plane(path).astype(np.float64)
    if array.min() < 0 or array.max() > maximum:
        raise ValueError(f'{path}: intensity outside [0, {maximum}]; set --intensity-max explicitly')
    return array / maximum
