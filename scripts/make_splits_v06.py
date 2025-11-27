#!/usr/bin/env python3
import json
import random
from pathlib import Path

manifest = json.loads(Path('data/program_manifest_v06.json').read_text())
programs = [entry['name'] for entry in manifest]
programs.sort()
random.seed(42)
random.shuffle(programs)

n = len(programs)
train = programs[: int(0.7 * n)]
val = programs[int(0.7 * n): int(0.85 * n)]
test = programs[int(0.85 * n):]

split = {
    'train_programs': train,
    'val_programs': val,
    'test_programs': test
}
Path('splits').mkdir(parents=True, exist_ok=True)
Path('splits/v06.json').write_text(json.dumps(split, indent=2))
print('Wrote splits/v06.json with', len(train), len(val), len(test))
