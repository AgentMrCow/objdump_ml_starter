#!/usr/bin/env python3
import argparse, json, pathlib
from elftools.elf.elffile import ELFFile
from elftools.elf.enums import ENUM_ST_INFO_TYPE
from elftools.elf.constants import SH_FLAGS


def collect_exec_ranges(elffile):
    ranges = []
    for sec in elffile.iter_sections():
        flags = sec.header.get("sh_flags", 0)
        if not (flags & SH_FLAGS.SHF_ALLOC):
            continue
        if not (flags & SH_FLAGS.SHF_EXECINSTR):
            continue
        start = int(sec["sh_addr"])
        size = int(sec.data_size)
        if size <= 0:
            continue
        ranges.append((start, start + size))
    return ranges


def in_exec(start, exec_ranges):
    if start <= 0:
        return False
    if not exec_ranges:
        return True
    return any(lo <= start < hi for lo, hi in exec_ranges)


def collect_symtab_funcs(elffile, exec_ranges):
    out = []
    for sec in elffile.iter_sections():
        if sec.header['sh_type'] == 'SHT_SYMTAB':
            for sym in sec.iter_symbols():
                t = sym['st_info']['type']
                if t == 'STT_FUNC' or t == ENUM_ST_INFO_TYPE['STT_FUNC']:
                    if sym.entry.get('st_shndx') == 'SHN_UNDEF':
                        continue
                    start = int(sym['st_value'])
                    if not in_exec(start, exec_ranges):
                        continue
                    size = int(sym['st_size'])
                    end = start + size if size else None
                    name = sym.name
                    out.append({"start": start, "end": end, "name": name})
    return out


def collect_dwarf_funcs(elffile, exec_ranges):
    out = []
    if not elffile.has_dwarf_info():
        return out
    di = elffile.get_dwarf_info()
    for cu in di.iter_CUs():
        for die in cu.iter_DIEs():
            if die.tag == 'DW_TAG_subprogram':
                lowpc = die.attributes.get('DW_AT_low_pc')
                highpc = die.attributes.get('DW_AT_high_pc')
                if not lowpc:
                    continue
                start = int(lowpc.value)
                if not in_exec(start, exec_ranges):
                    continue
                end = None
                if highpc:
                    # class may be 'address' or 'constant'
                    hp = highpc.value
                    if highpc.form == 'DW_FORM_addr':
                        end = int(hp)
                    else:
                        end = start + int(hp)
                name_attr = die.attributes.get('DW_AT_name')
                name = name_attr.value.decode('utf-8', errors='ignore') if name_attr else ""
                out.append({"start": start, "end": end, "name": name})
    return out

def unify(funcs):
    # prefer entries with both start and end; dedupe by start
    by_start = {}
    for f in funcs:
        s = f["start"]
        if s not in by_start:
            by_start[s] = f
        else:
            if by_start[s].get("end") is None and f.get("end") is not None:
                by_start[s] = f
            elif f.get("name") and not by_start[s].get("name"):
                by_start[s]["name"] = f["name"]
    # best-effort: fill missing ends by sorting
    items = sorted(by_start.values(), key=lambda x: x["start"])
    for i in range(len(items)-1):
        if items[i].get("end") is None:
            items[i]["end"] = items[i+1]["start"]
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True, help="output JSON with {start,end,name}")
    args = ap.parse_args()
    with open(args.bin, "rb") as f:
        ef = ELFFile(f)
        exec_ranges = collect_exec_ranges(ef)
        funcs = collect_symtab_funcs(ef, exec_ranges) + collect_dwarf_funcs(ef, exec_ranges)
        uni = unify(funcs)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as w:
            json.dump(uni, w, indent=2)
    print(f"Wrote {args.out} with {len(uni)} functions.")

if __name__ == "__main__":
    main()
