#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

def is_branch(ins):
    m = ins.get('mnemonic','')
    return m.startswith('j') or m in {'loop','loope','loopne','jecxz','jrcxz'}


def build_cfg(instrs):
    # naïve block graph: treat every branch target and fallthrough as edges
    addr_to_idx = {ins['addr']: i for i, ins in enumerate(instrs)}
    preds = {i:set() for i in range(len(instrs))}
    succs = {i:set() for i in range(len(instrs))}
    for i, ins in enumerate(instrs):
        outs = []
        for t in ins.get('xrefs_out', []):
            if isinstance(t, int) and t in addr_to_idx:
                outs.append(addr_to_idx[t])
        if i+1 < len(instrs):
            outs.append(i+1)  # fallthrough
        for o in outs:
            succs[i].add(o)
            preds[o].add(i)
    return preds, succs, addr_to_idx


def augment_file(path):
    data = json.load(open(path))
    instrs = data['instrs']
    preds, succs, addr_to_idx = build_cfg(instrs)
    for idx, ins in enumerate(instrs):
        ins['bb_preds'] = len(preds[idx])
        ins['bb_succs'] = len(succs[idx])
        block_len = 1
        j = idx+1
        while j < len(instrs) and preds[j] == {idx} and len(preds[j])==1:
            block_len +=1; j+=1
        ins['bb_len'] = block_len
        bytes_len = len(ins.get('bytes',[]))
        ins['byte_entropy'] = sum(ins.get('bytes',[]))/bytes_len if bytes_len else 0
    json.dump(data, open(path,'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asm_glob', required=True)
    args = ap.parse_args()
    import glob
    paths = glob.glob(args.asm_glob)
    for p in paths:
        augment_file(p)
    print(f"augmented {len(paths)} files with cfg features")

if __name__=='__main__':
    main()
