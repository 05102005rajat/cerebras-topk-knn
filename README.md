# Top-K k-NN on the Cerebras Wafer-Scale Engine

A CSL (Cerebras Software Language) kernel that computes the K nearest
neighbors (squared L2) of a query vector against a database sharded across a
P×P grid of processing elements (PEs), submitted for the Cerebras Kernel
Challenge ("Wafer-Scale Top-K k-NN") — a timed take-home used in their
technical hiring loop. Compiled with `cslc --arch=wse2` and executed on
Cerebras's own WSE-2 functional simulator via `cs_python`; every PE computes
its local candidates independently and the result is reduced across the
fabric with zero host involvement between input load and output readback.

**Start time:** 2026-04-29 (logged per SPEC §7; challenge is time-boxed to 72
hours). Submitted 2026-05-01.

## Key results

- **Local top-K, ~32× fewer element-ops than selection sort.** Extraction
  uses a bounded max-heap + heapsort: `O((rows_per_pe − K)·log K + K·log K)`
  against `O(rows_per_pe · K)` for the naive approach. On the `k_large` grader
  case (`K = rows_per_pe = 256`) that Big-O reduces to a ~32× win
  (`256·log₂256 = 2048` vs. `256·256 = 65536`).
- **P-way merge, ~8× fewer compares than re-heaping.** Each PE's local top-K
  arrives at the reduction stage already sorted, so the row/column merge is a
  linear-min scan over P sorted runs — `O(K·P)` against `O(K·P·log K)` for
  re-heaping the combined input. On `k_large` (K=256, P=2), `log₂K = 8`: the
  row-merge drops from ~4,096 compares to ~512.
- **Compute-bound, not routing-bound.** Worst-case fabric traffic is ≤1,024
  u32 words per edge (K=256, P=2, dist+idx on both axes) against ~12k FPU
  element-ops per PE for distance compute alone — the local heap/merge work
  comfortably masks fabric latency (full accounting in `DESIGN.md` §3).
- **6/6 grader cases exercised end-to-end** through the real Cerebras SDK
  toolchain, not a mock: `run_all.sh` compiles and simulates baseline,
  `K=1` (argmin), `K=256` (large-K, SRAM-pressure), `N` not divisible by `P²`
  (uneven sharding), all-identical rows, and duplicate rows at non-adjacent
  indices — see [Test matrix](#test-matrix) below.
- **Deterministic under arbitrary wavelet arrival order.** Every comparison,
  at every stage (local heap, row merge, column merge), goes through a single
  strict total order `lex_lt` on `(distance, global_row_index)` — the
  composability argument (local-min → row-min → global-min ≡ global top-K)
  is in `DESIGN.md` §4.

AI-assisted development was explicitly permitted by the challenge spec (§7,
"you may use any AI assistant, we encourage it"); the design memo — not code
authorship — is what's graded for understanding.

## Architecture

- **Grid:** P×P PEs, each owning `ceil(N / P²)` rows of the database shard
  (`D_shard`) plus a broadcast copy of the query vector `q`.
- **Distance compute:** squared L2 per row, vectorized across
  `rows_per_pe` lanes via CSL DSDs (`compute_distances`) — 3 DSD ops
  (`fsubs`/`fmuls`/`fadds`) per input dimension.
- **Reduction:** two SDK `collectives_2d` gathers per axis (distance +
  index), row-then-column — `mpi_x.gather` into column 0, merge to a row
  top-K, `mpi_y.gather` into row 0, merge to the global top-K at PE(0,0).
  No custom wavelet routing or packing; the whole reduction is a chained
  SPMD task list (`kernel_main → x_dist_done → x_idx_done → y_dist_done →
  y_idx_done`) fired by each collective's completion callback.
- **Padding:** on uneven shards, rows past `valid_rows` are rewritten
  in-kernel to a `(f32::MAX, u32::MAX)` sentinel so they always sort to the
  back of the local top-K.
- **Host driver (`run.py`):** memcpy's the shard/query/offsets in, launches
  `kernel_main`, reads `final_dist`/`final_idx` back from PE(0,0), and
  compares against a NumPy oracle (`reference.py`, `np.lexsort` on
  `(index, distance)`) with `atol=rtol=1e-3` on distances and bit-exact
  index match.

## Test matrix

All 6 cases are grader-defined (`tests/test_correctness.py` in the challenge
repo) and reproduced locally via `run_all.sh`:

| Case | N | d | K | P | Stresses |
|---|---|---|---|---|---|
| `baseline` | 2048 | 32 | 16 | 4 | correctness at the intended working size |
| `k_eq_1` | 1024 | 32 | 1 | 2 | argmin edge case |
| `k_large` | 1024 | 16 | 256 | 2 | large-K SRAM pressure, heap/merge cost dominates |
| `uneven` | 1009 | 32 | 16 | 4 | N not divisible by P² — last PE's shard is short |
| `all_equal` | 1024 | 16 | 16 | 2 | every row identical — tie-break must return `[0..K-1]` |
| `duplicates` | 1024 | 16 | 8 | 2 | exact-duplicate rows at non-adjacent indices |

## Layout

```
solution/
├── layout.csl        # P×P grid, exports + collectives_2d wiring
├── pe_program.csl    # per-PE kernel: distances, local top-K, x/y gathers, merges
├── run.py            # host driver (memcpy, launch, oracle compare, PASS marker)
├── commands.sh       # baseline cslc + cs_python invocation
├── run_all.sh        # convenience: compile+run all 6 grader cases
├── reference.py      # oracle (copy of challenge reference.py; see note below)
├── DESIGN.md         # one-page memo (routing, algo, BW, tie-break, future)
└── README.md         # this file
```

`reference.py` is included in the submission because the cs_python
container only bind-mounts `PWD` inside the SIF — `../reference.py` from
the challenge tree is invisible to the kernel host driver. Keeping a copy
beside `run.py` makes the submission self-contained.

## Build + run (baseline)

```bash
bash commands.sh
```

The grader (`tests/test_correctness.py --submission=<this dir>`) compiles
`layout.csl` once per case with case-specific `--params` and runs
`cs_python run.py --name <build_dir> --case <case>`; it looks for the
`PASS: <case>` marker in stdout. `run_all.sh` runs all 6 cases in sequence
the same way.

## Where to look

| Function | File | Does |
|---|---|---|
| `compute_distances` | `pe_program.csl` | row-vectorized squared-L2 via DSDs |
| `topk_select` | `pe_program.csl` | bounded max-heap + heapsort local top-K |
| `merge_sorted_runs` | `pe_program.csl` | P-way linear-min merge of pre-sorted K-runs |
| `lex_lt` | `pe_program.csl` | the one comparator everything routes through |
| `kernel_main` … `y_idx_done` | `pe_program.csl` | task chain driving the two collective gathers |

See `DESIGN.md` for the full architectural rationale, including the fabric
bandwidth accounting and the tie-break composability proof.
