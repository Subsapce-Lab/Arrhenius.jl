#!/usr/bin/env python3
"""Run one case group through the production Cantera/native gate using one Julia session.

Use separate invocations for independent groups; the candidate repository is read-only.
The harness and references are read from the supplied validation root.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="NH3 validation harness root")
    parser.add_argument("--native", type=Path, required=True, help="candidate Arrhenius worktree")
    parser.add_argument("--manifest", type=Path, required=True, help="nh3_set_manifest.py output")
    parser.add_argument("--ids", type=Path, required=True, help="case IDs for this group, one per line")
    parser.add_argument("--out", type=Path, required=True, help="shared output directory; case IDs must be unique")
    args = parser.parse_args()

    root, native, out = args.root.resolve(), args.native.resolve(), args.out.resolve()
    py, jl = os.environ["PY"], os.environ["JL"]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", LC_ALL="C")
    cases = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    with args.manifest.open(newline="") as stream:
        rows = {row["id"]: row for row in csv.DictReader(stream)}
    missing = [case for case in cases if case not in rows]
    if missing:
        raise SystemExit(f"IDs missing from manifest: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    conflicts = [case for case in cases if (out / f"{case}.json").exists()]
    if conflicts:
        raise SystemExit(f"refusing to overwrite existing results: {conflicts}")

    server = subprocess.Popen(
        [jl, f"--project={native}", str(root / "h" / "nh3_native_server.jl")],
        cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT, text=True,
    )
    try:
        for case in cases:
            row = rows[case]
            spec, reference = row["spec"], row["reference"]
            result = out / f"{case}.json"
            with tempfile.TemporaryDirectory(prefix=f"pointfix-{case}-") as scratch:
                scratch_path = Path(scratch)
                ct_out = scratch_path / "cantera.npz"
                native_out = scratch_path / "native.npz"
                marker = scratch_path / "native.done"
                ct_result = subprocess.run(
                    [py, str(root / "h" / "nh3_cantera.py"), spec, str(ct_out),
                     "--batches=1", "--per-batch=1"], cwd=root, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    timeout=900, check=False,
                )
                if ct_result.returncode:
                    raise RuntimeError(f"Cantera failed for {case}: {ct_result.stdout[-3000:]}")
                request = f"{spec}\t{native_out}\t3\t1\t{marker}\n"
                assert server.stdin is not None
                server.stdin.write(request)
                server.stdin.flush()
                deadline = time.monotonic() + 900
                while not marker.exists():
                    if server.poll() is not None:
                        raise RuntimeError(f"native server exited with {server.returncode} on {case}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"native timed out on {case} after 900 s")
                    time.sleep(0.2)
                status = marker.read_text().strip()
                if status != "DONE":
                    raise RuntimeError(f"native failed for {case}: {status}")
                gate = subprocess.run(
                    [py, str(root / "h" / "nh3_gate.py"), spec, reference,
                     str(ct_out), str(native_out), str(result)], cwd=root, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    timeout=120, check=False,
                )
                if gate.returncode or not result.is_file():
                    raise RuntimeError(f"production gate failed for {case}: {gate.stdout[-3000:]}")
                print(f"BRANCHCHECK_DONE {case} gate={result}", flush=True)
    finally:
        if server.poll() is None:
            if server.stdin is not None:
                server.stdin.write("QUIT\n")
                server.stdin.flush()
            server.wait(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
