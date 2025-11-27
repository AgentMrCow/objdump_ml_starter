#!/usr/bin/env python3
import json
import random
from pathlib import Path

TOTAL = 1200
BASE_DIR = Path('samples/auto_v06')
BASE_DIR.mkdir(parents=True, exist_ok=True)
manifest = []
random.seed(42)

helper_templates = [
    """static int helper_{idx}(int n) {{
    int acc = {start};
    for (int i = 0; i < n; ++i) {{
        acc {op}= (i % {mod}) + {add};
    }}
    return acc;
}}
""",
    """static int helper_{idx}(int n) {{
    if (n <= 1) return n + {add};
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {{
        int t = a + b + {add};
        a = b;
        b = t;
    }}
    return b + {start};
}}
""",
    """static int helper_{idx}(int n) {{
    int acc = {start};
    for (int i = 1; i <= n; ++i) {{
        acc += (n / (i | 1)) - {add};
    }}
    return acc;
}}
""",
    """static int helper_{idx}(int n) {{
    int acc = 0;
    for (int i = 0; i < n; ++i) {{
        for (int j = 0; j < {mod}; ++j) {{
            acc += (i ^ j) & {mask};
        }}
    }}
    return acc + {add};
}}
"""
]

main_template = """#include <stdio.h>\n#include <stdint.h>\n{helpers}
{extra_funcs}
int main(void) {{
    int total = 0;
    for (int i = 0; i < {loop_bound}; ++i) {{
        total += helper_{first}(i + {delta});
    }}
    printf(\"%s: %d\\n\", "{name}", total);
    return total & 0xff;
}}
"""

lib_template = """#include <stdint.h>\n#include <stddef.h>\n{helpers}
{extra_funcs}
int {name}_api(int n) {{
    if (n < 0) n = -n;
    int result = helper_{first}(n % {loop_bound});
    return result ^ {xor_mask};
}}
"""

extra_func_template = """static void blend_{idx}(int *arr, size_t len) {{
    for (size_t i = 1; i < len; ++i) {{
        arr[i] = (arr[i-1] + arr[i]) ^ {mask};
    }}
}}
"""

for idx in range(TOTAL):
    prog_name = f"auto_v06_{idx:04d}"
    prog_dir = BASE_DIR / prog_name
    prog_dir.mkdir(parents=True, exist_ok=True)
    helper_blocks = []
    helper_count = random.randint(2, 5)
    helper_ids = []
    for h in range(helper_count):
        tpl = random.choice(helper_templates)
        params = {
            'idx': h,
            'start': random.randint(1, 13),
            'op': random.choice(['+', '-', '^']),
            'mod': random.randint(3, 9),
            'add': random.randint(1, 7),
            'mask': random.randint(3, 31)
        }
        helper_blocks.append(tpl.format(**params))
        helper_ids.append(h)
    extra_funcs = []
    for e in range(random.randint(1, 3)):
        extra_funcs.append(extra_func_template.format(idx=e, mask=random.randint(1, 63)))
    has_main = (idx % 4 != 0)
    first_helper = helper_ids[0]
    if has_main:
        code = main_template.format(
            helpers="\n".join(helper_blocks),
            extra_funcs="\n".join(extra_funcs),
            loop_bound=random.randint(5, 25),
            delta=random.randint(1, 5),
            first=first_helper,
            name=prog_name
        )
    else:
        code = lib_template.format(
            helpers="\n".join(helper_blocks),
            extra_funcs="\n".join(extra_funcs),
            first=first_helper,
            loop_bound=random.randint(7, 30),
            xor_mask=random.randint(1, 255),
            name=prog_name
        )
    src_path = prog_dir / "prog.c"
    src_path.write_text(code)
    manifest.append({
        "name": prog_name,
        "src": str(src_path),
        "has_main": has_main,
        "language": "c"
    })

manifest_path = Path('data/programs_v06.json')
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"Generated {TOTAL} programs -> {manifest_path}")
