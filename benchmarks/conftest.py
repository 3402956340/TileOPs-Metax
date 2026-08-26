import gc

import pytest
import torch

# Imported for its side effect: arming the guard that keeps flag_gems from
# reaching torch's op registry before vllm. See benchmarks.baselines.
import benchmarks.baselines  # noqa: F401
from benchmarks.benchmark_base import BenchmarkReport
from benchmarks.report import _bench_results
from benchmarks.xfail_whitelist import MACA_XFAIL_PREFIXES, MACA_XFAILS
from tileops.utils import is_maca


def _normalized_benchmark_nodeid(item: pytest.Item) -> str:
    nodeid = item.nodeid
    if nodeid.startswith("benchmarks/"):
        return nodeid
    if nodeid.startswith("ops/"):
        return f"benchmarks/{nodeid}"
    return nodeid


def _is_fp8_e4m3_benchmark(item: pytest.Item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    return callspec.params.get("dtype") == torch.float8_e4m3fn


def pytest_make_parametrize_id(config, val, argname):
    """Render the values pytest would otherwise collect as `shape0`, `dtype0`.

    A case id is the workload's name everywhere it is read later — the nightly
    report, the published page, the perf history key. This covers the values
    with no readable repr; it does not invent a name for the case, which is the
    author's job (see .claude/domain-rules/benchmark.md).
    """
    if isinstance(val, torch.dtype):
        return str(val).removeprefix("torch.")
    if isinstance(val, tuple) and val and all(isinstance(v, int) for v in val):
        return "x".join(str(v) for v in val)
    if isinstance(val, bool):
        name = argname
        for prefix in ("has_", "is_", "use_", "with_", "num_", "n_"):
            name = name.removeprefix(prefix)
        name = name.replace("_", "")
        return name if val else f"no{name}"
    return None


# Set by the recorder, not measurements.
_NOT_A_MEASUREMENT = frozenset({"tag", "op", "op_module"})


def _prop(value) -> str:
    """Format one measurement for the XML.

    Significant digits rather than fixed decimals: rates across the suite span
    six orders of magnitude, and a sub-microsecond kernel loses several percent
    to four decimal places.
    """
    if isinstance(value, (bool, int)):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    return str(value)


def _emit(item, tag: str, entry: dict) -> None:
    """Publish every measurement an implementation recorded.

    Generic over the keys: a measurement added to the benchmark layer reaches
    the XML, and the consumers that parse `<tag>_<metric>`, without a change
    here. Hand-listing them is how the report came to publish a quantity the
    benchmark had stopped comparing.
    """
    for key, value in entry.items():
        if key in _NOT_A_MEASUREMENT or value is None:
            continue
        item.user_properties.append((f"{tag}_{key}", _prop(value)))


def _release_cuda_cache_after_case() -> None:
    """Drop per-case Python references and cached CUDA blocks between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_maca_xfails(items: list[pytest.Item]) -> None:
    """Mark exact-node known failures on MACA benchmark runs."""
    if not is_maca():
        return

    for item in items:
        nodeid = _normalized_benchmark_nodeid(item)
        reason = MACA_XFAILS.get(nodeid)
        if reason is None:
            reason = next(
                (
                    prefix_reason
                    for prefix, prefix_reason in MACA_XFAIL_PREFIXES.items()
                    if nodeid.startswith(prefix)
                ),
                None,
            )
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=f"MACA: {reason}", strict=False))


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def pytest_sessionstart(session):
    BenchmarkReport.clear()


def pytest_sessionfinish(session, exitstatus):
    BenchmarkReport.dump("profile_run.log")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    fp8_e4m3_skip = pytest.mark.skip(
        reason=(
            "Skipped under tilelang 0.1.9: fp8 e4m3 benchmark fails due to "
            "lowering regression; re-enable when fp8 e4m3 benchmarks run "
            "cleanly against current tilelang."
        )
    )

    for item in items:
        nodeid = _normalized_benchmark_nodeid(item)
        path = nodeid.split("::", 1)[0]

        if path == "benchmarks/ops/bench_elementwise_fp8.py" and _is_fp8_e4m3_benchmark(item):
            item.add_marker(fp8_e4m3_skip)

    _apply_maca_xfails(items)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """After bench test execution, attach perf data to the item as properties."""
    _bench_results.entries = []
    try:
        yield
        entries = getattr(_bench_results, "entries", [])
        if not entries:
            return

        # Separate tileops entry (tag starts with "tileops") from baselines.
        tileops_entry = None
        baseline_entries = []
        for e in entries:
            if e["tag"].startswith("tileops"):
                if tileops_entry is None:
                    tileops_entry = e
            else:
                baseline_entries.append(e)

        if tileops_entry:
            item.user_properties.append(("op", tileops_entry["op"]))
            if "op_module" in tileops_entry:
                item.user_properties.append(("op_module", tileops_entry["op_module"]))
            tag = tileops_entry["tag"]
            if tag != "tileops" and tag.startswith("tileops_"):
                item.user_properties.append(("tileops_variant", tag[len("tileops_") :]))
            _emit(item, "tileops", tileops_entry)

        # Every baseline is written under its own tag. The first also uses the
        # unprefixed legacy names that scripts/nightly_report.py reads.
        for idx, be in enumerate(baseline_entries):
            tag = be["tag"]
            _emit(item, tag, be)
            if idx == 0:
                item.user_properties.append(("baseline_tag", tag))
                _emit(item, "baseline", be)
            if not tileops_entry:
                continue
            # Ratios compare device_busy_ms: two implementations need not have
            # the same number of gaps between kernels.
            tl = tileops_entry.get("device_busy_ms", 0)
            bl = be.get("device_busy_ms", 0)
            if tl > 0 and bl > 0:
                item.user_properties.append((f"{tag}_ratio", f"{bl / tl:.4f}"))
                if idx == 0:
                    item.user_properties.append(("baseline_ratio", f"{bl / tl:.4f}"))
    finally:
        _bench_results.entries = []
        _release_cuda_cache_after_case()
