#!/usr/bin/env python3
"""Run benchmark ops in two phases for nightly CI.

Phase A (bulk): one ``pytest <root>`` session with ``--ignore`` for paths in
``benchmarks.conftest.SERIAL_NODE_BENCH_PATHS``.

Phase B (serial): for each serial path, collect parametrized node ids and run
each in its own pytest process to avoid GPU memory accumulation.

Outputs a merged JUnit XML and a concatenated ``profile_run.log``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.conftest import SERIAL_NODE_BENCH_PATHS  # noqa: E402


def _normalize_collected_nodeid(nodeid: str) -> str:
    """Match ``benchmarks/conftest._normalized_benchmark_nodeid`` for strings."""
    if nodeid.startswith("benchmarks/"):
        return nodeid
    if nodeid.startswith("ops/"):
        return f"benchmarks/{nodeid}"
    return nodeid


def _collect_nodeids(target: str) -> list[str]:
    """Return collected pytest node ids for *target* (file or directory)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            target,
            "--collect-only",
            "-q",
            "--no-header",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    if result.returncode not in (0, 5):
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"pytest collection failed for {target} (exit {result.returncode})")

    nodeids: list[str] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or " collected" in line or line.startswith("no tests"):
            continue
        if "::" not in line:
            continue
        nodeid = _normalize_collected_nodeid(line.split()[0])
        if nodeid not in seen:
            seen.add(nodeid)
            nodeids.append(nodeid)
    return nodeids


def _run_pytest(
    targets: list[str],
    *,
    junit: Path | None,
    extra_args: list[str] | None = None,
) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    if junit is not None:
        cmd.extend([f"--junit-xml={junit}"])
    if extra_args:
        cmd.extend(extra_args)
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=_REPO_ROOT).returncode


def _merge_junit_xml(input_files: list[Path], output: Path) -> None:
    merged = ET.Element("testsuites")
    for path in input_files:
        if not path.is_file():
            continue
        root = ET.parse(path).getroot()
        if root.tag == "testsuite":
            merged.append(root)
            continue
        for child in root:
            if child.tag == "testsuite":
                merged.append(child)
    ET.ElementTree(merged).write(output, encoding="utf-8", xml_declaration=True)


def _append_profile_log(source: Path, dest: Path) -> None:
    if not source.is_file():
        return
    with dest.open("a", encoding="utf-8") as out, source.open(encoding="utf-8") as inp:
        content = inp.read()
        out.write(content)
        if content and not content.endswith("\n"):
            out.write("\n")


def _path_under_root(path: Path, root_path: Path) -> bool:
    """True when *path* is covered by *root* (directory or single file)."""
    if root_path.is_file():
        return path == root_path
    try:
        path.relative_to(root_path)
        return True
    except ValueError:
        return False


def _resolve_serial_paths(root: str) -> list[str]:
    """Serial paths that exist under *root* (warn on missing entries)."""
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = (_REPO_ROOT / root_path).resolve()

    resolved: list[str] = []
    for rel in sorted(SERIAL_NODE_BENCH_PATHS):
        path = (_REPO_ROOT / rel).resolve()
        if not path.is_file():
            print(f"::warning::SERIAL_NODE_BENCH_PATHS entry not found: {rel}", flush=True)
            continue
        if not _path_under_root(path, root_path):
            print(
                f"::warning::SERIAL_NODE_BENCH_PATHS entry outside --root: {rel}",
                flush=True,
            )
            continue
        resolved.append(rel)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="benchmarks/ops",
        help="Benchmark tree root passed to bulk phase (default: benchmarks/ops)",
    )
    parser.add_argument(
        "--junit-out",
        default="bench_results.xml",
        help="Merged JUnit XML output path (default: bench_results.xml)",
    )
    parser.add_argument(
        "--profile-log",
        default="profile_run.log",
        help="Final concatenated profile log path (default: profile_run.log)",
    )
    parser.add_argument(
        "--junit-dir",
        default=".bench_junit_parts",
        help="Temporary directory for per-phase JUnit fragments",
    )
    args = parser.parse_args(argv)

    serial_paths = _resolve_serial_paths(args.root)
    junit_dir = (_REPO_ROOT / args.junit_dir).resolve()
    junit_dir.mkdir(parents=True, exist_ok=True)
    for old in junit_dir.glob("*.xml"):
        old.unlink()

    profile_final = _REPO_ROOT / args.profile_log
    profile_final.unlink(missing_ok=True)
    profile_tmp = junit_dir / "profile_run.concat.log"
    profile_tmp.unlink(missing_ok=True)

    junit_parts: list[Path] = []
    failed = False

    # --- Phase A: bulk (root minus serial files) ---
    ignore_args = [f"--ignore={path}" for path in serial_paths]
    bulk_junit = junit_dir / "bulk.xml"
    print(f"=== Phase A: bulk {args.root} ({len(serial_paths)} file(s) ignored) ===", flush=True)
    bulk_rc = _run_pytest([args.root], junit=bulk_junit, extra_args=ignore_args)
    _append_profile_log(_REPO_ROOT / "profile_run.log", profile_tmp)
    (_REPO_ROOT / "profile_run.log").unlink(missing_ok=True)
    # pytest exits 5 when "no tests collected"; tolerate this in bulk phase
    # because all benchmarks may be routed to Phase B.
    if bulk_rc not in (0, 5):
        failed = True
        print(f"::warning::Phase A bulk benchmarks exited with code {bulk_rc}", flush=True)
    if bulk_junit.is_file():
        junit_parts.append(bulk_junit)

    # --- Phase B: serial per node for listed files ---
    print(f"=== Phase B: serial nodes for {len(serial_paths)} file(s) ===", flush=True)
    for bench_file in serial_paths:
        nodeids = _collect_nodeids(bench_file)
        print(f"--- {bench_file}: {len(nodeids)} node(s) ---", flush=True)
        if not nodeids:
            print(f"::warning::No benchmark nodes collected for {bench_file}", flush=True)
            continue
        safe_name = bench_file.replace("/", "_").removesuffix(".py")
        for idx, nodeid in enumerate(nodeids):
            case_junit = junit_dir / f"serial_{safe_name}_{idx}.xml"
            print(f"[{idx + 1}/{len(nodeids)}] {nodeid}", flush=True)
            case_rc = _run_pytest([nodeid], junit=case_junit)
            _append_profile_log(_REPO_ROOT / "profile_run.log", profile_tmp)
            (_REPO_ROOT / "profile_run.log").unlink(missing_ok=True)
            if case_rc != 0:
                failed = True
                print(f"::warning::failed: {nodeid} (exit {case_rc})", flush=True)
            if case_junit.is_file():
                junit_parts.append(case_junit)

    junit_out = _REPO_ROOT / args.junit_out
    if junit_parts:
        _merge_junit_xml(junit_parts, junit_out)
        print(f"Merged {len(junit_parts)} JUnit fragment(s) -> {junit_out}", flush=True)
    else:
        print("::warning::No JUnit fragments produced", flush=True)
        failed = True

    if profile_tmp.is_file():
        profile_tmp.replace(profile_final)
    else:
        print("::warning::profile_run.log not produced", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
