#@category Export
#@menupath Tools.ExportFunctions

import csv
import os

args = getScriptArgs()
if len(args) < 2:
    raise RuntimeError("Usage: ghidra_export_functions.py <bin_path> <out_dir>")

bin_path = args[0]
out_dir = args[1]
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
stem = os.path.basename(bin_path)
out_path = os.path.join(out_dir, stem + ".csv")

program = getCurrentProgram()
fm = program.getFunctionManager()
funcs = list(fm.getFunctions(True))
funcs.sort(key=lambda f: int(f.getEntryPoint().getOffset()))

with open(out_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["binary", "start", "end", "name"])
    for func in funcs:
        start = int(func.getEntryPoint().getOffset())
        body = func.getBody()
        end_addr = body.getMaxAddress()
        end = int(end_addr.getOffset()) + 1
        writer.writerow([stem, start, end, func.getName()])

print("Exported %d functions to %s" % (len(funcs), out_path))
