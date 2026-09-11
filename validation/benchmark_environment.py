"""Host checks shared by the paired calculation benchmarks."""
import hashlib
import os
from pathlib import Path
import platform
import subprocess


def cpu_brand():
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def host_metadata():
    return {"cpu": cpu_brand(), "platform": platform.platform(),
            "kernel_release": platform.release(), "system": platform.system(),
            "machine": platform.machine(), "logical_cpus": os.cpu_count()}


def matches_target(host, target):
    if target == "wsl":
        return (host.get("system") == "Linux"
                and "microsoft" in host.get("kernel_release", "").lower())
    if target == "apple-m4":
        return host.get("system") == "Darwin" and "Apple M4" in host.get("cpu", "")
    raise ValueError(f"unknown benchmark target: {target}")


def cantera_library_hashes(module_path):
    """Hash loaded Cantera shared libraries on Linux, installed dylibs on macOS."""
    paths = set()
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            path = line.split()[-1]
            if path.startswith("/") and "libcantera" in Path(path).name:
                paths.add(Path(path).resolve())
    else:
        for parent in Path(module_path).resolve().parents:
            lib = parent / "lib"
            paths.update(p.resolve() for p in lib.glob("libcantera_shared*.dylib"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
