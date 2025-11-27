#!/usr/bin/env python3
import json
from pathlib import Path
import shutil

REPO = Path('tmp/rosettacode')
TASK_DIR = REPO / 'Task'
OUT_DIR = Path('samples/real_v06')
OUT_DIR.mkdir(parents=True, exist_ok=True)
manifest = []
count = 0

LANG_EXTS = {
    'C': '.c',
}

for task in sorted(TASK_DIR.iterdir()):
    if not task.is_dir():
        continue
    lang_dir = task / 'C'
    if not lang_dir.exists():
        continue
    for src_file in lang_dir.rglob('*.c'):
        prog_name = f"rosetta_v06_{count:05d}_{task.name.replace(' ', '_')}_{src_file.stem}"
        dest_dir = OUT_DIR / prog_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / 'prog.c'
        shutil.copy(src_file, dest_path)
        code = dest_path.read_text(errors='ignore')
        has_main = 'main(' in code
        manifest.append({
            'name': prog_name,
            'src': str(dest_path),
            'has_main': has_main,
            'language': 'c'
        })
        count += 1

manifest_path = Path('data/program_manifest_v06.json')
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"Collected {count} C sources -> {manifest_path}")
