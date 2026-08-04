"""Exact-node allowlist for known benchmark failures on MetaX MACA."""

_UNSUPPORTED_ARCHITECTURE = "kernel is not supported on the current MACA architecture"
_RUNTIME_LAUNCH_ERROR = "known MACA runtime launch error"
_COMPILATION_FAILURE = "known MACA benchmark compilation failure"
_BENCHMARK_API_MISMATCH = "benchmark uses an incompatible cumulative Op constructor"
_AUTOTUNE_FAILURE = "no benchmark configuration compiles and validates successfully"


# FIXME(staged-rollout): quarantine the current MetaX benchmark failures by exact node ID.
#
# Broken invariant: every collected TileOps benchmark runs on the MetaX test runner.
# Why: the backend and several benchmark call sites still have known compatibility gaps.
# Cleanup: remove each entry as soon as its node passes consistently on the MetaX runner.
_MACA_XFAIL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        _UNSUPPORTED_ARCHITECTURE,
        (
            "benchmarks/ops/attention/bench_deepseek_dsa_decode.py::test_dsa_decode_bench[single-batch-mainstream-float16]",
            "benchmarks/ops/attention/bench_deepseek_dsa_decode.py::test_dsa_decode_bench[longer-kv-lower-topk-float16]",
        ),
    ),
    (
        _RUNTIME_LAUNCH_ERROR,
        (
        ),
    ),
    (
        _COMPILATION_FAILURE,
        (
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_logical_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_and_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_or_manifest_bench[cnn-feat-broadcast-bool]",
            "benchmarks/ops/bench_elementwise_manifest.py::test_bitwise_xor_manifest_bench[cnn-feat-broadcast-bool]",
        ),
    ),
    (
        _BENCHMARK_API_MISMATCH,
        (
        ),
    ),
    (
        _AUTOTUNE_FAILURE,
        (
        ),
    ),
)

MACA_XFAILS = {
    nodeid: reason
    for reason, nodeids in _MACA_XFAIL_GROUPS
    for nodeid in nodeids
}

if len(MACA_XFAILS) != sum(len(nodeids) for _, nodeids in _MACA_XFAIL_GROUPS):
    raise ValueError("duplicate node ID in the MACA benchmark xfail allowlist")
