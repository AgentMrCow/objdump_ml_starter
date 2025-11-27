#!/usr/bin/env python3
import os
import pathlib
import re
import json
import hashlib
import subprocess
import textwrap

URL_LIST = pathlib.Path('data/program_urls_v06.txt')
DEST_ROOT = pathlib.Path('samples/v06_sources')
DEST_ROOT.mkdir(parents=True, exist_ok=True)
manifest = []

if not URL_LIST.exists():
    raise SystemExit('data/program_urls_v06.txt missing')

with URL_LIST.open() as f:
    urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

for idx, url in enumerate(urls):
    name = re.sub(r'[^A-Za-z0-9]+', '_', url.split('/')[-1]) or f'prog_{idx}'
    prog_name = f'real_v06_{idx:04d}_{name}'
    prog_dir = DEST_ROOT / prog_name
    prog_dir.mkdir(parents=True, exist_ok=True)
    dest = prog_dir / 'prog.c'
    print(f'Fetching {url} -> {dest}')
    try:
        subprocess.run(['curl', '-L', url, '-o', str(dest)], check=True)
    except subprocess.CalledProcessError as e:
        print('Failed:', url, e)
        continue
    data = dest.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    manifest.append({
        'name': prog_name,
        'src': str(dest),
        'has_main': True,
        'sha256': sha,
        'source_url': url
    })

manifest_path = pathlib.Path('data/program_manifest_v06.json')
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f'Wrote manifest -> {manifest_path} with {len(manifest)} entries')
