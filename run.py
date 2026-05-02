#!/usr/bin/env cs_python
"""Host driver for the Top-K k-NN kernel.

Invoked by the grader as `cs_python run.py --name <build_dir> --case <case>`.

Steps:
  1. Re-derive the test-case inputs from reference.py (deterministic via seed).
  2. Compute the NumPy oracle.
  3. Memcpy each PE's D shard, broadcast q to every PE, set per-PE
     `index_offset` and `valid_rows`.
  4. Launch the kernel; wait for completion (every PE unblocks the cmd stream).
  5. Read `final_dist` and `final_idx` back from PE(0,0).
  6. Compare against the oracle: indices must match bit-identically; distances
     must be within atol=rtol=1e-3. Print `PASS: <case>` on success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Locate reference.py — the grader puts it at a parent of our submission dir.
# Don't resolve() __file__: if our submission lives at a symlink, resolve()
# would jump out of the challenge tree and we'd miss reference.py.
_HERE = Path(__file__).parent.absolute()
_search = [_HERE, *_HERE.parents]
for _candidate in _search:
    if (_candidate / "reference.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError(f"reference.py not found in {_search}")

from reference import ALL_CASES, topk_reference  # noqa: E402

from cerebras.sdk.runtime.sdkruntimepybind import (  # noqa: E402
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)


# Grader's snake_case keys → reference.py display names.
_CASE_KEY_TO_DISPLAY = {
    "baseline":   "baseline",
    "k_eq_1":     "k=1",
    "k_large":    "k=256",
    "uneven":     "uneven",
    "all_equal":  "all-equal",
    "duplicates": "duplicates",
}


def _resolve_case(case_key: str) -> dict:
    if case_key not in _CASE_KEY_TO_DISPLAY:
        raise ValueError(f"unknown case: {case_key!r}")
    target = _CASE_KEY_TO_DISPLAY[case_key]
    for maker in ALL_CASES:
        case = maker()
        if case["name"] == target:
            return case
    raise RuntimeError(f"case {target!r} not produced by any maker")


def _shard(D: np.ndarray, P: int, rows_per_pe: int):
    """Return per-PE shard, index offsets, and valid-row counts.

    Distribution is contiguous: PE i owns rows [i·rows_per_pe, (i+1)·rows_per_pe).
    The last PE may have fewer valid rows when N is not divisible — those slots
    are zero-padded here, and the kernel rewrites them to (INF, U32MAX) so they
    sort to the back of the local top-K.
    """
    N, d = D.shape
    total_pes = P * P

    padded = np.zeros((total_pes * rows_per_pe, d), dtype=np.float32)
    padded[:N] = D

    D_per_pe = padded.reshape(total_pes, rows_per_pe, d)
    offsets = (np.arange(total_pes) * rows_per_pe).astype(np.uint32)

    valid = np.zeros(total_pes, dtype=np.uint32)
    for i in range(total_pes):
        start = i * rows_per_pe
        end = min((i + 1) * rows_per_pe, N)
        valid[i] = max(0, end - start)

    return D_per_pe, offsets, valid


def _to_hwl(arr_per_pe: np.ndarray, P: int) -> np.ndarray:
    """Reshape (P*P, ...) → (P, P, ...) so memcpy_h2d ROW_MAJOR maps PE flat
    index py*P+px to data[py, px]."""
    return arr_per_pe.reshape(P, P, *arr_per_pe.shape[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="cslc -o output dir")
    parser.add_argument("--case", required=True, help="case key (snake_case)")
    parser.add_argument("--cmaddr", default=None, help="optional fabric address")
    args = parser.parse_args()

    case = _resolve_case(args.case)
    D = np.ascontiguousarray(case["D"], dtype=np.float32)
    q = np.ascontiguousarray(case["q"], dtype=np.float32)
    K = int(case["K"])
    P = int(case["P"])
    N, d = D.shape
    rows_per_pe = -(-N // (P * P))  # ceil(N / P²)

    # Oracle.
    ref_idx, ref_dist = topk_reference(D, q, K, squared=True)

    # Shard inputs.
    D_per_pe, offsets, valid = _shard(D, P, rows_per_pe)
    D_hwl       = _to_hwl(D_per_pe, P).astype(np.float32, copy=False)
    q_hwl       = np.broadcast_to(q, (P, P, d)).astype(np.float32, copy=True)
    offsets_hwl = _to_hwl(offsets.reshape(-1, 1), P).astype(np.uint32, copy=False)
    valid_hwl   = _to_hwl(valid.reshape(-1, 1), P).astype(np.uint32, copy=False)

    # Launch. cmaddr=None routes to the simulator (verified gemv-collectives_2d pattern).
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

    sym_D      = runner.get_id("D_shard")
    sym_q      = runner.get_id("q")
    sym_offset = runner.get_id("index_offset")
    sym_valid  = runner.get_id("valid_rows")
    sym_fin_d  = runner.get_id("final_dist")
    sym_fin_i  = runner.get_id("final_idx")

    runner.load()
    runner.run()

    u32 = MemcpyDataType.MEMCPY_32BIT
    rmo = MemcpyOrder.ROW_MAJOR

    runner.memcpy_h2d(sym_D, D_hwl.ravel(), 0, 0, P, P, rows_per_pe * d,
                      streaming=False, data_type=u32, nonblock=False, order=rmo)
    runner.memcpy_h2d(sym_q, q_hwl.ravel(), 0, 0, P, P, d,
                      streaming=False, data_type=u32, nonblock=False, order=rmo)
    runner.memcpy_h2d(sym_offset, offsets_hwl.ravel(), 0, 0, P, P, 1,
                      streaming=False, data_type=u32, nonblock=False, order=rmo)
    runner.memcpy_h2d(sym_valid, valid_hwl.ravel(), 0, 0, P, P, 1,
                      streaming=False, data_type=u32, nonblock=False, order=rmo)

    runner.launch("kernel_main", nonblock=False)

    out_dist = np.zeros(K, dtype=np.float32)
    out_idx  = np.zeros(K, dtype=np.uint32)
    runner.memcpy_d2h(out_dist, sym_fin_d, 0, 0, 1, 1, K,
                      streaming=False, data_type=u32, nonblock=False, order=rmo)
    runner.memcpy_d2h(out_idx,  sym_fin_i, 0, 0, 1, 1, K,
                      streaming=False, data_type=u32, nonblock=False, order=rmo)

    runner.stop()

    out_idx_i32 = out_idx.astype(np.int32)
    if not np.array_equal(out_idx_i32, ref_idx):
        print(f"FAIL: {args.case} — index mismatch")
        print(f"  expected: {ref_idx[:min(K, 16)].tolist()}")
        print(f"  got:      {out_idx_i32[:min(K, 16)].tolist()}")
        sys.exit(1)
    if not np.allclose(out_dist, ref_dist, atol=1e-3, rtol=1e-3):
        print(f"FAIL: {args.case} — distance mismatch")
        print(f"  expected: {ref_dist[:min(K, 16)].tolist()}")
        print(f"  got:      {out_dist[:min(K, 16)].tolist()}")
        sys.exit(1)

    print(f"PASS: {args.case}")


if __name__ == "__main__":
    main()
