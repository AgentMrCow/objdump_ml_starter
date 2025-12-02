#!/usr/bin/env python3
import json
import argparse
import pathlib

def augment(asm):
    instrs = asm['instrs']
    for i, ins in enumerate(instrs):
        # basic block start if previous ins is branch/ret or this is first
        if i == 0:
            ins['bb_start'] = 1
        else:
            prev = instrs[i-1]['mnemonic']
            if prev.startswith('ret') or prev.startswith('jmp') or prev.startswith('j'):
                ins['bb_start'] = 1
            else:
                ins['bb_start'] = 0
        # byte entropy proxy
        b = ins.get('bytes', [])
        ins['byte_sum'] = sum(b)
        ins['byte_len'] = len(b)
        # prologue hint: stack pointer writes nearby
        if 'rsp' in ins.get('ops','') and ('sub' in ins.get('mnemonic','') or 'add' in ins.get('mnemonic','')):
            ins['rsp_touch'] = 1
        else:
            ins['rsp_touch'] = 0
    return asm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asm_glob', required=True)
    args = ap.parse_args()
    import glob
    paths = glob.glob(args.asm_glob)
    for p in paths:
        data = json.load(open(p))
        data = augment(data)
        json.dump(data, open(p,'w'), indent=2)
    print(f"Augmented {len(paths)} asm files")

if __name__ == '__main__':
    main()
