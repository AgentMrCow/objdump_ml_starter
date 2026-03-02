#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from elftools.elf.constants import SH_FLAGS


def load_global_dict(path: Path):
    mapping = {}
    with path.open('r') as f:
        for idx, line in enumerate(f):
            mapping[line.strip()] = idx
    return mapping


def get_exec_sections(elf: ELFFile):
    sections = []
    for section in elf.iter_sections():
        if not (section['sh_flags'] & SH_FLAGS.SHF_ALLOC) or section.data_size == 0:
            continue
        if section['sh_flags'] & SH_FLAGS.SHF_EXECINSTR:
            sections.append((section['sh_addr'], section.data_size, section.data()))
    return sections


def map_function_offsets(elf: ELFFile, exec_sections):
    func_addrs = []
    for section in elf.iter_sections():
        if not isinstance(section, SymbolTableSection):
            continue
        if section['sh_entsize'] == 0:
            continue
        for symbol in section.iter_symbols():
            if symbol.entry['st_info']['type'] == 'STT_FUNC' and symbol.entry['st_shndx'] != 'SHN_UNDEF':
                func_addrs.append(symbol['st_value'])
    mapped = set()
    offset = 0
    for sec_addr, sec_size, _ in exec_sections:
        for faddr in func_addrs:
            if sec_addr <= faddr < sec_addr + sec_size:
                mapped.add(offset + (faddr - sec_addr))
        offset += sec_size
    return mapped


def convert_binary(bin_path: Path, out_path: Path, global_dict, superset_disa, handle_ins):
    with bin_path.open('rb') as f:
        elf = ELFFile(f)
        arch = elf.elfclass
        exec_sections = get_exec_sections(elf)
        if not exec_sections:
            raise RuntimeError(f"no executable sections in {bin_path}")
        code = b"".join(sec[2] for sec in exec_sections)
        func_offsets = map_function_offsets(elf, exec_sections)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = []
    cnt = 0
    with out_path.open('w') as out:
        for ins in superset_disa(arch, code, base_offset=0):
            label = 'S' if ins.address in func_offsets else 'N'
            buf.append(','.join(handle_ins(ins, global_dict)) + f',{label},' + hex(ins.address))
            cnt += 1
            if cnt >= 1000:
                out.write("\n".join(buf) + "\n")
                buf = []
                cnt = 0
        if buf:
            out.write("\n".join(buf) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bins_list', required=True, help='list of stripped binaries (will map to _sym)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--disa_root', default='tmp/external/Disa/Disa_task1_2')
    args = ap.parse_args()

    disa_root = Path(args.disa_root)
    sys.path.insert(0, str(disa_root))

    from superset_disa import superset_disa
    from get_feature_from_ins import handle_ins

    global_dict = load_global_dict(disa_root / 'global_dict.txt')

    bins = [line.strip() for line in Path(args.bins_list).read_text().splitlines() if line.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bin_path in bins:
        bin_path = Path(bin_path)
        sym_path = bin_path.with_name(bin_path.name.replace('_stripped', '_sym'))
        if not sym_path.exists():
            print(f"[skip] missing sym binary: {sym_path}")
            continue
        out_path = out_dir / f"{sym_path.name}.txt"
        if out_path.exists():
            print(f"[skip] exists: {out_path}")
            continue
        print(f"convert {sym_path} -> {out_path}")
        convert_binary(sym_path, out_path, global_dict, superset_disa, handle_ins)


if __name__ == '__main__':
    main()
