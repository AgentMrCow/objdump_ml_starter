#!/usr/bin/env python3
import csv
from pathlib import Path
rows=[]
for path in Path('out/ghidra_compare_v06d_O3.tsv').read_text().splitlines()[1:]:
    if not path.strip():
        continue
    parts=path.split('\t')
    file, agree, miss, extra = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
    rows.append((file, miss, extra, agree))
rows.sort(key=lambda x: (x[1]+x[2]), reverse=True)
print('Top 10 worst (miss+extra):')
for r in rows[:10]:
    print(r)
