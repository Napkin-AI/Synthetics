"""One auditable results bundle per CLI invocation, with exact scored masks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from urllib.parse import quote

import numpy as np
from PIL import Image


def utc_now():
    return datetime.now(timezone.utc)


def slug(value):
    return re.sub(r'[^\w.-]+', '_', str(value), flags=re.UNICODE).strip('._')[:64] or 'run'


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in str(value):
        raise ValueError(f'Unsafe artifact path component: {value}')
    return Path(*(quote(part, safe='-_.') for part in path.parts))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class RunArtifacts:
    def __init__(self, args, command):
        self.root = args.results_dir.resolve()
        report_dir, segmentation_dir = self.root/'report', self.root/'segmentation'
        report_dir.mkdir(parents=True, exist_ok=True)
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        config_name = args.task_config.stem if args.task_config else args.mode
        info = args.run_info or args.synthetic.name
        base = f'{utc_now():%Y%m%d_%H%M%S_%fUTC}_{slug(config_name)}_{slug(info)}'
        for index in range(1000):
            self.name = base if index == 0 else f'{base}_{index}'
            self.directory = segmentation_dir/self.name
            self.report = report_dir/f'{self.name}.json'
            if self.report.exists():
                continue
            try:
                self.directory.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError('Could not allocate a unique results run')
        for domain in ('real', 'synthetic'):
            (self.directory/domain).mkdir()
        self.state = {'run_name': self.name, 'status': 'running', 'started_at_utc': utc_now().isoformat(),
                      'cwd': str(Path.cwd()), 'argv': command, 'saved_mask_pairs': {'real': 0, 'synthetic': 0},
                      'saved_originals': {'real': 0, 'synthetic': 0}}
        self.state['parameters'] = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
        # Only variables affecting this Python/numerical execution, never the entire environment.
        env = {key: os.environ[key] for key in ('TORCH_DEVICE_BACKEND_AUTOLOAD', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS') if key in os.environ}
        self.state['environment_overrides'] = env
        if os.name == 'nt':
            lines = [f'cd /d "{Path.cwd()}"', *[f'set "{k}={v}"' for k, v in env.items()], subprocess.list2cmdline(command)]
        else:
            lines = [f'cd -- {shlex.quote(str(Path.cwd()))}',
                     shlex.join(['env', *[f'{k}={v}' for k, v in env.items()], *command])]
        (self.directory/'command.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        self.update_state()

    def update_state(self):
        atomic_json(self.directory/'run.json', self.state)

    def save_original(self, domain, key, source):
        if domain not in ('real', 'synthetic'):
            raise ValueError('Unknown artifact domain')
        source = Path(source)
        # Preserve source filename/format, including Lucchi names before key aliasing.
        relative = safe_relative(key).parent / safe_relative(source.name)
        path = self.directory/domain/'original'/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Duplicate original output: {path}')
        shutil.copyfile(source, path)
        checksum = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                checksum.update(block)
        self.state['saved_originals'][domain] += 1
        return {'path': str(path.relative_to(self.root)), 'sha256': checksum.hexdigest()}

    def save_masks(self, domain, key, class_name, prediction, target):
        if domain not in ('real', 'synthetic'):
            raise ValueError('Unknown artifact domain')
        relative = safe_relative(key)
        # Append, do not replace: image keys can contain dots (training0028.png_...).
        relative = relative.parent/(relative.name+'.png')
        folder = safe_relative(class_name)
        saved = {}
        for kind, mask in (('prediction', prediction), ('target', target)):
            base = self.directory/domain/('predictions' if kind == 'prediction' else 'targets')
            path = base/folder/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f'Duplicate segmentation output: {path}')
            Image.fromarray(np.asarray(mask, dtype=np.uint8)*255).save(path)
            saved[kind] = str(path.relative_to(self.root))
            saved[kind+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.state['saved_mask_pairs'][domain] += 1
        return saved

    def finish(self, result):
        result['artifacts'] = {'results_dir': str(self.root), 'run_name': self.name,
                               'report': str(self.report), 'segmentation': str(self.directory),
                               'command': str(self.directory/'command.txt'),
                               'mask_paths_relative_to': 'results_dir',
                               'mask_encoding': 'binary PNG uint8: background 0, foreground 255',
                               'saved_mask_pairs': self.state['saved_mask_pairs'],
                               'saved_originals': self.state['saved_originals'],
                               'original_encoding': 'byte-for-byte copy of input image, original format and bit depth'}
        atomic_json(self.report, result)
        self.state.update(status=result['status'], finished_at_utc=utc_now().isoformat(), report=str(self.report))
        self.update_state()

    def fail(self, error):
        self.state.update(status='failed', finished_at_utc=utc_now().isoformat(),
                          error=f'{type(error).__name__}: {error}')
        self.update_state()
