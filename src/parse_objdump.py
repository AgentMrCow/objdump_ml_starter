#!/usr/bin/env python3
import json, re, subprocess, argparse, sys, pathlib

INSTR_RE = re.compile(r'^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F ]+)\s+\t([.\w-]+)\s*(.*)$')
LABEL_RE = re.compile(r'^([0-9a-fA-F]+)\s+<([^>]+)>:\s*$')

def run_objdump(path: str) -> str:
    try:
        out = subprocess.check_output(["objdump", "-d", "-M", "intel", path], stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        print(e.output.decode("utf-8", errors="replace"), file=sys.stderr)
        raise

def parse(text: str):
    instrs = []
    labels = {}  # addr -> label name
    for line in text.splitlines():
        m = LABEL_RE.match(line)
        if m:
            addr = int(m.group(1), 16)
            labels[addr] = m.group(2)
            continue
        m = INSTR_RE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        bytes_str = m.group(2).strip()
        mnemonic = m.group(3).strip()
        ops = m.group(4).strip()
        bs = [int(b, 16) for b in bytes_str.split() if b]
        instrs.append({
            "addr": addr,
            "bytes": bs,
            "mnemonic": mnemonic,
            "ops": ops,
            "line": line
        })
    # compute xrefs (very simple immediate call/jmp target parsing)
    for ins in instrs:
        tgt = None
        if ins["mnemonic"].startswith("call") or ins["mnemonic"].startswith("jmp"):
            # look for hex address in ops
            m = re.search(r'([0-9a-fA-Fx]+)', ins["ops"])
            if m:
                s = m.group(1)
                try:
                    if s.lower().startswith("0x"):
                        tgt = int(s, 16)
                    else:
                        tgt = int(s, 16)
                except Exception:
                    tgt = None
        ins["xrefs_out"] = [tgt] if tgt is not None else []
    # build xrefs_in
    x_in = {}
    for ins in instrs:
        for t in ins.get("xrefs_out", []):
            x_in.setdefault(t, 0)
            x_in[t] += 1
    for ins in instrs:
        ins["xrefs_in"] = x_in.get(ins["addr"], 0)
    return {"instrs": instrs, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text = run_objdump(args.bin)
    data = parse(text)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.out} with {len(data['instrs'])} instructions.")

if __name__ == "__main__":
    main()
