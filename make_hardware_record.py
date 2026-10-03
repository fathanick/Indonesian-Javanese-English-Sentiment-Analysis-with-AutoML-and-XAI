"""Write validation_evidence/hardware.txt from the live machine plus saved artifacts.

Captured on the machine that produced runs/encoders-20260924, so the record is a
measurement rather than a restatement of the manuscript.
Run: python3 make_hardware_record.py

INPUTS ARE NOT REDISTRIBUTED. This reads two artifacts of the reported run that
are not published with this repository: runs/encoders-20260924/preflight.json and
validation_evidence/omp_runtime_defect/hang_sample.txt. It also invokes macOS
`sysctl` and `sw_vers`, so it is macOS-only as written.
"""
import json
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE / "validation_evidence"
EV.mkdir(exist_ok=True)


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def sysctl(key):
    return sh("sysctl", "-n", key)


KEYS = ["hw.model", "hw.memsize", "hw.ncpu", "hw.physicalcpu", "hw.logicalcpu",
        "hw.perflevel0.logicalcpu", "hw.perflevel1.logicalcpu",
        "hw.perflevel0.name", "hw.perflevel1.name",
        "kern.osproductversion", "machdep.cpu.brand_string", "hw.optional.arm64"]

pf = json.loads((HERE / "runs/encoders-20260924/preflight.json").read_text())
hang = (EV / "omp_runtime_defect/hang_sample.txt").read_text(errors="replace")
os_line = next((l.strip() for l in hang.splitlines() if "OS Version:" in l), "n/a")

mem = int(sysctl("hw.memsize"))
lines = [
    "Hardware and operating-system record",
    f"Captured: {sh('date', '+%Y-%m-%d %H:%M:%S %Z')} on the machine that produced "
    "runs/encoders-20260924",
    "",
    "sw_vers",
    sh("sw_vers"),
    "",
    "sysctl",
]
for k in KEYS:
    lines.append(f"  {k:<28} = {sysctl(k)}")
lines += [
    "",
    "derived",
    f"  hw.memsize                 = {mem} bytes = {mem / 2**30:.0f} GiB unified memory",
    f"  python -c platform.platform() = {platform.platform()}",
    "",
    "cross-check against saved run artifacts",
    f"  runs/encoders-20260924/preflight.json  platform   = {pf['platform']}",
    f"  runs/encoders-20260924/preflight.json  python     = {pf['python']}",
    f"  omp_runtime_defect/hang_sample.txt     {os_line}",
    "",
    "Notes",
    "  The operator was a single MacBook Pro. hw.model Mac16,8 is the M4 Pro 14-inch",
    "  model; 12 logical cores = 8 performance + 4 efficiency.",
    "  macOS 26.5.2 (build 25F84) is the version under which the development, fit,",
    "  evaluate, explain and report stages ran; it is also the version recorded in the",
    "  deadlock sampling report, so the two independent records agree.",
]
(EV / "hardware.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
