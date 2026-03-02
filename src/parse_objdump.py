#!/usr/bin/env python3
import json, re, subprocess, argparse, sys, pathlib
from features import compute_reachable_addrs

try:
    import capstone
    from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_REG_RIP
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection
    from elftools.elf.constants import SH_FLAGS
    HAVE_RECURSIVE = True
except Exception:
    HAVE_RECURSIVE = False

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
    # compute xrefs (very simple immediate branch/call target parsing)
    for ins in instrs:
        tgt = None
        mnem = ins["mnemonic"]
        if mnem.startswith("call") or mnem.startswith("j"):
            # look for hex address in ops (prefer 0x... or long hex)
            m = re.search(r'(0x[0-9a-fA-F]+|[0-9a-fA-F]{4,})', ins["ops"])
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
    reachable = sorted(compute_reachable_addrs(instrs, labels))
    reachable_set = set(reachable)
    for ins in instrs:
        ins["reachable"] = 1 if ins["addr"] in reachable_set else 0
    return {"instrs": instrs, "labels": labels, "reachable_addrs": reachable}


def _get_exec_sections(elf):
    sections = []
    for section in elf.iter_sections():
        if not (section['sh_flags'] & SH_FLAGS.SHF_ALLOC) or section.data_size == 0:
            continue
        if section['sh_flags'] & SH_FLAGS.SHF_EXECINSTR:
            sections.append((section['sh_addr'], section.data_size, section.data()))
    return sections


def _collect_labels(elf):
    labels = {}
    for section in elf.iter_sections():
        if not isinstance(section, SymbolTableSection):
            continue
        if section['sh_entsize'] == 0:
            continue
        for symbol in section.iter_symbols():
            if symbol.entry['st_info']['type'] == 'STT_FUNC' and symbol.entry['st_shndx'] != 'SHN_UNDEF':
                labels[int(symbol['st_value'])] = symbol.name
    return labels


def _in_exec(addr, ranges):
    for start, end in ranges:
        if start <= addr < end:
            return True
    return False


def parse_recursive(path: str):
    if not HAVE_RECURSIVE:
        raise RuntimeError("capstone/pyelftools not available for recursive disassembly")

    with open(path, 'rb') as f:
        elf = ELFFile(f)
        arch = elf.elfclass
        exec_sections = _get_exec_sections(elf)
        all_sections = []
        for section in elf.iter_sections():
            if not (section['sh_flags'] & SH_FLAGS.SHF_ALLOC) or section.data_size == 0:
                continue
            all_sections.append((section['sh_addr'], section.data_size, section.data()))
        if not exec_sections:
            return {"instrs": [], "labels": {}, "reachable_addrs": []}
        labels = _collect_labels(elf)
        entry = int(elf.header.get('e_entry', 0))

    ranges = [(start, start + size) for start, size, _ in exec_sections]
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64 if arch == 64 else capstone.CS_MODE_32)
    md.detail = True
    ptr_size = 8 if arch == 64 else 4

    def decode_one(addr):
        for start, size, data in exec_sections:
            if start <= addr < start + size:
                off = addr - start
                chunk = data[off:off + 16]
                insts = md.disasm(chunk, addr, count=1)
                try:
                    return next(insts)
                except StopIteration:
                    return None
        return None

    def read_ptr(addr):
        for start, size, data in all_sections:
            if start <= addr < start + size:
                off = addr - start
                if off + ptr_size > size:
                    return None
                return int.from_bytes(data[off:off + ptr_size], "little", signed=False)
        return None

    def extract_rip_mem_base(ins):
        for op in ins.operands:
            if op.type == X86_OP_MEM and op.mem.base == X86_REG_RIP:
                return int(ins.address + ins.size + op.mem.disp)
        return None

    def extract_imm(ins):
        for op in ins.operands:
            if op.type == X86_OP_IMM:
                return int(op.imm)
        return None

    def find_jumptable_targets(ins_by_addr, instrs):
        if not instrs:
            return set()
        addrs_sorted = sorted(ins_by_addr.keys())
        prev_map = {addr: addrs_sorted[i - 1] if i > 0 else None for i, addr in enumerate(addrs_sorted)}
        targets = set()
        for item in instrs:
            if not item["mnemonic"].startswith("jmp") or item["xrefs_out"]:
                continue
            ins = ins_by_addr.get(item["addr"])
            if ins is None:
                continue
            base = None
            base = extract_rip_mem_base(ins)
            if base is None:
                prev_addr = prev_map.get(item["addr"])
                prev = ins_by_addr.get(prev_addr) if prev_addr is not None else None
                if prev is not None and prev.mnemonic in {"lea", "mov"}:
                    base = extract_rip_mem_base(prev)
                    if base is None and prev.mnemonic == "mov":
                        imm = extract_imm(prev)
                        base = imm if imm is not None else None
                    if base is None and prev.mnemonic == "mov":
                        rip = extract_rip_mem_base(prev)
                        if rip is not None:
                            base = read_ptr(rip)
            if base is None:
                continue
            # scan a small table window for executable pointers
            for i in range(32):
                val = read_ptr(base + i * ptr_size)
                if val is None:
                    break
                if _in_exec(val, ranges):
                    targets.add(val)
        return targets

    seeds = {entry} if entry else set()
    for addr in labels:
        if _in_exec(addr, ranges):
            seeds.add(addr)
    # fall back to section starts if no seeds
    if not seeds:
        for start, _, _ in exec_sections:
            seeds.add(start)

    visited = set()
    instrs = []
    ins_by_addr = {}
    stack = list(seeds)

    def walk_stack():
        while stack:
            addr = stack.pop()
            if addr in visited:
                continue
            ins = decode_one(addr)
            if ins is None:
                visited.add(addr)
                continue
            visited.add(addr)
            ins_by_addr[int(ins.address)] = ins
            mnem = ins.mnemonic
            ops = ins.op_str
            bs = list(ins.bytes)

            direct_target = None
            if mnem.startswith("call") or mnem.startswith("jmp") or (mnem.startswith("j") and not mnem.startswith("jmp")):
                for op in ins.operands:
                    if op.type == X86_OP_IMM:
                        direct_target = int(op.imm)
                        break

            xrefs_out = [direct_target] if isinstance(direct_target, int) else []
            instrs.append({
                "addr": int(ins.address),
                "bytes": bs,
                "mnemonic": mnem,
                "ops": ops,
                "line": f"{ins.address:x}: {mnem} {ops}",
                "xrefs_out": xrefs_out,
            })

            fallthrough = int(ins.address + ins.size)
            if mnem.startswith("ret") or mnem in {"hlt", "ud2"}:
                continue
            if mnem.startswith("jmp") and direct_target is not None:
                if _in_exec(direct_target, ranges):
                    stack.append(direct_target)
                continue
            if mnem.startswith("jmp"):
                continue
            if mnem.startswith("j") and not mnem.startswith("jmp"):
                if direct_target is not None and _in_exec(direct_target, ranges):
                    stack.append(direct_target)
                if _in_exec(fallthrough, ranges):
                    stack.append(fallthrough)
                continue
            if mnem.startswith("call"):
                if _in_exec(fallthrough, ranges):
                    stack.append(fallthrough)
                if direct_target is not None and _in_exec(direct_target, ranges):
                    stack.append(direct_target)
                continue
            if _in_exec(fallthrough, ranges):
                stack.append(fallthrough)

    walk_stack()
    # try to recover indirect jump targets via jump-table heuristic
    for _ in range(2):
        new_targets = find_jumptable_targets(ins_by_addr, instrs)
        new_targets = [t for t in new_targets if t not in visited]
        if not new_targets:
            break
        stack.extend(new_targets)
        walk_stack()

    instrs.sort(key=lambda x: x["addr"])
    for ins in instrs:
        ins["reachable"] = 1

    # build xrefs_in
    x_in = {}
    for ins in instrs:
        for t in ins.get("xrefs_out", []):
            x_in.setdefault(t, 0)
            x_in[t] += 1
    for ins in instrs:
        ins["xrefs_in"] = x_in.get(ins["addr"], 0)

    reachable = sorted({ins["addr"] for ins in instrs})
    return {"instrs": instrs, "labels": labels, "reachable_addrs": reachable}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["objdump", "recursive", "hybrid"], default="hybrid")
    args = ap.parse_args()
    if args.mode == "recursive":
        if not HAVE_RECURSIVE:
            print("capstone/pyelftools unavailable; falling back to objdump.", file=sys.stderr)
            text = run_objdump(args.bin)
            data = parse(text)
        else:
            data = parse_recursive(args.bin)
    elif args.mode == "hybrid":
        text = run_objdump(args.bin)
        data = parse(text)
        if HAVE_RECURSIVE:
            rec = parse_recursive(args.bin)
            rec_reachable = set(rec.get("reachable_addrs", []))
            if rec_reachable:
                for ins in data["instrs"]:
                    ins["reachable"] = 1 if (ins.get("reachable", 0) or ins["addr"] in rec_reachable) else 0
            data["reachable_addrs"] = sorted(rec_reachable)
        else:
            print("capstone/pyelftools unavailable; hybrid uses objdump-only reachability.", file=sys.stderr)
    else:
        text = run_objdump(args.bin)
        data = parse(text)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.out} with {len(data['instrs'])} instructions.")

if __name__ == "__main__":
    main()
