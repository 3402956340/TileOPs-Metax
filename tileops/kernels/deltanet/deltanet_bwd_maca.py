"""DeltaNet backward MACA path: smem-safe dh_recurrence + tiled compute_w_u_bwd."""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .deltanet_bwd import _bwd_parallel_tl

__all__ = [
    "DeltaNetBwdMACAKernel",
]


@functools.lru_cache(maxsize=32)
def _dh_recurrence_bwd_tl_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Sequential backward dh recurrence with V-tile corrections (MACA smem-safe).

    Carry state lives in shared ``dh_carry [dim_k, dim_v]``; each V-tile uses
    ``T.copy`` slices (gated-style) to avoid fragment layout conflicts.
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = 32
    sub_dim_v = dim_v // BV

    assert dim_v % BV == 0, "dim_v must be divisible by BV"

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_recurrence_bwd_kernel_maca(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], accum_dtype),
            dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                dh_carry = T.alloc_shared([dim_k, dim_v], accum_dtype)
                dh_loc = T.alloc_shared([dim_k, BV], accum_dtype)
                dh_tile = T.alloc_shared([dim_k, BV], accum_dtype)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                k_dh_shared = T.alloc_shared([block_C, BV], dtype)

                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dP_tmp = T.alloc_fragment([block_C, dim_k], accum_dtype)
                wk_dh_frag = T.alloc_fragment([dim_k, BV], accum_dtype)

                for i, j in T.Parallel(dim_k, dim_v):
                    dh_carry[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(w[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], w_c, disable_tma=True)

                    T.clear(dP_frag)

                    for v0 in T.serial(0, sub_dim_v):
                        v_off = v0 * BV
                        T.copy(
                            v_new[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, v_off : v_off + BV],
                            v_new_c, disable_tma=True,
                        )
                        T.copy(
                            dh_local[bid, hid, t_bwd, :, v_off : v_off + BV],
                            dh_loc, disable_tma=True,
                        )

                        T.copy(
                            dh_carry[:, v_off : v_off + BV],
                            dh_tile, disable_tma=True,
                        )
                        T.copy(dh_tile, dh_buf, disable_tma=True)

                        T.clear(du_corr_frag)
                        T.gemm(k_c, dh_buf, du_corr_frag)
                        T.copy(
                            du_corr_frag,
                            du_corr[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, v_off : v_off + BV],
                            disable_tma=True,
                        )
                        T.copy(du_corr_frag, k_dh_shared)

                        T.clear(dP_tmp)
                        T.gemm(v_new_c, dh_buf, dP_tmp, transpose_B=True)
                        for n, kk in T.Parallel(block_C, dim_k):
                            dP_frag[n, kk] += dP_tmp[n, kk]

                        T.clear(wk_dh_frag)
                        T.gemm(w_c, k_dh_shared, wk_dh_frag, transpose_A=True)
                        for i, j in T.Parallel(dim_k, BV):
                            dh_tile[i, j] = (
                                dh_tile[i, j] + dh_loc[i, j] - wk_dh_frag[i, j]
                            )
                        T.copy(
                            dh_tile,
                            dh_carry[:, v_off : v_off + BV],
                            disable_tma=True,
                        )

                    for n, kk in T.Parallel(block_C, dim_k):
                        dk_corr[bid, hid, t_bwd * block_C + n, kk] = dP_frag[n, kk]

        return dh_recurrence_bwd_kernel_maca

    return _func


def _compute_dw_corr(
    du_corr: torch.Tensor,
    S: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Per-chunk dw_corr = du_corr @ S^T (S is boundary state at chunk start)."""
    batch, head, seq_len, dim_v = du_corr.shape
    dim_k = S.shape[-2]
    num_chunks = seq_len // chunk_size
    du_c = du_corr.float().reshape(batch, head, num_chunks, chunk_size, dim_v)
    s_c = S[:, :, :num_chunks].float()
    dw_corr = torch.einsum("bhcnd,bhckd->bhcnk", du_c, s_c)
    return dw_corr.reshape(batch, head, seq_len, dim_k).to(du_corr.dtype)


@torch.library.custom_op("tileops::deltanet_bwd_kernel_maca", mutates_args=())
def _deltanet_bwd_wrapped_kernel_maca(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    num_stages: int, threads: int,
    parallel_threads: int, recurrence_threads: int,
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, beta: torch.Tensor,
    S: torch.Tensor,
    Aw: torch.Tensor, Au: torch.Tensor,
    w: torch.Tensor, u: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .compute_w_u_bwd_maca import compute_w_u_bwd_tl_maca

    bwd_parallel_fn = _bwd_parallel_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(parallel_threads)
    dh_recurrence_bwd_fn = _dh_recurrence_bwd_tl_maca(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(num_stages, recurrence_threads)
    wu_bwd_fn = compute_w_u_bwd_tl_maca(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(num_stages, threads)

    dq, dk_partial, dw, du_partial, v_new, dh_local =         bwd_parallel_fn(do, q, k, w, u, S)
    dk_corr, du_corr =         dh_recurrence_bwd_fn(k, w, v_new, dh_local)

    du = du_partial + du_corr
    dw_total = dw - _compute_dw_corr(du_corr, S, chunk_size)
    dk_wu, dv, dbeta = wu_bwd_fn(dw_total, du, Aw, Au, k, v, beta)
    dk = dk_partial + dk_corr + dk_wu
    return dq, dk, dv, dbeta


@_deltanet_bwd_wrapped_kernel_maca.register_fake
def _deltanet_bwd_wrapped_kernel_maca_fake(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    num_stages: int, threads: int,
    parallel_threads: int, recurrence_threads: int,
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, beta: torch.Tensor,
    S: torch.Tensor,
    Aw: torch.Tensor, Au: torch.Tensor,
    w: torch.Tensor, u: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty(batch, head, seq_len, dim_k, dtype=q.dtype, device=q.device)
    dk = torch.empty_like(dq)
    dv = torch.empty(batch, head, seq_len, dim_v, dtype=v.dtype, device=v.device)
    dbeta = torch.empty(batch, head, seq_len, dtype=beta.dtype, device=beta.device)
    return dq, dk, dv, dbeta


class DeltaNetBwdMACAKernel(Kernel):
    """DeltaNet backward kernel for MACA (smem-safe tiled path)."""

    supported_archs: list[int] = [80, 89, 90]
    # Placeholder so init_config(tune=True) invokes custom autotune() below.
    autotune_configs: list[dict] = [{}]

    def __init__(
        self,
        batch: int,
        head: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        threads = 256 if self.chunk_size >= 64 else 128
        return {
            "num_stages": 1,
            "threads": threads,
            "parallel_threads": threads,
            "recurrence_threads": threads,
        }

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune each sub-kernel; MACA keeps num_stages=1 for pipelined stages."""
        from tilelang.autotuner import autotune as tl_autotune

        from .compute_w_u_bwd_maca import compute_w_u_bwd_tl_maca

        B, H, S, BC = self.batch, self.head, self.seq_len, self.chunk_size
        DK, DV, dt = self.dim_k, self.dim_v, self.dtype_str

        parallel_configs = [{"threads": t} for t in [128, 256]]
        print(f"Autotuning bwd_parallel ({len(parallel_configs)} configs)...")
        parallel_jit = _bwd_parallel_tl(B, H, S, BC, DK, DV, dt)
        tuned_parallel = tl_autotune(configs=parallel_configs, warmup=warmup, rep=rep)(parallel_jit)()
        parallel_best = tuned_parallel.config
        print(f"  Best: {parallel_best}")

        # MACA: pipelined recurrence/wu_bwd use extra smem; only stage count 1.
        recurrence_configs = [
            {"num_stages": 1, "threads": t}
            for t in [128, 256]
        ]
        print(f"Autotuning dh_recurrence_bwd_maca ({len(recurrence_configs)} configs)...")
        recurrence_jit = _dh_recurrence_bwd_tl_maca(B, H, S, BC, DK, DV, dt)
        tuned_recurrence = tl_autotune(
            configs=recurrence_configs, warmup=warmup, rep=rep,
        )(recurrence_jit)()
        recurrence_best = tuned_recurrence.config
        print(f"  Best: {recurrence_best}")

        wu_bwd_configs = [
            {"num_stages": 1, "threads": t}
            for t in [128, 256]
        ]
        print(f"Autotuning compute_w_u_bwd_maca ({len(wu_bwd_configs)} configs)...")
        wu_bwd_jit = compute_w_u_bwd_tl_maca(B, H, S, BC, DK, DV, dt)
        tuned_wu_bwd = tl_autotune(configs=wu_bwd_configs, warmup=warmup, rep=rep)(wu_bwd_jit)()
        wu_bwd_best = tuned_wu_bwd.config
        print(f"  Best: {wu_bwd_best}")

        self.config = {
            "num_stages": 1,
            "threads": wu_bwd_best["threads"],
            "parallel_threads": parallel_best["threads"],
            "recurrence_threads": recurrence_best["threads"],
        }
        print(f"DeltaNetBwdMACAKernel autotuned config: {self.config}")

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _deltanet_bwd_wrapped_kernel_maca(
            self.batch, self.head, self.seq_len, self.chunk_size,
            self.dim_k, self.dim_v, self.dtype_str,
            self.config.get("num_stages", 1), self.config.get("threads", 256),
            self.config.get("parallel_threads", 256),
            self.config.get("recurrence_threads", 256),
            do, q, k, v, beta, S, Aw, Au, w, u,
        )
