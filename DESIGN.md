# Top-K k-NN — Design Memo

## 1. Routing topology

P×P PE grid. Both reductions go through SDK's `<collectives_2d>`; no custom
routing or wavelet packing.

```
                                     y-gather root
                                          ↓
              col 0    col 1    col 2    col 3
            ┌────────┬────────┬────────┬────────┐
   row 0    │ (0,0)  │ (1,0)  │ (2,0)  │ (3,0)  │ ← x-gather root in each row
            ├────────┼────────┼────────┼────────┤
   row 1    │ (0,1)  │ (1,1)  │ (2,1)  │ (3,1)  │
            ├────────┼────────┼────────┼────────┤
   row 2    │ (0,2)  │ (1,2)  │ (2,2)  │ (3,2)  │
            ├────────┼────────┼────────┼────────┤
   row 3    │ (0,3)  │ (1,3)  │ (2,3)  │ (3,3)  │
            └────────┴────────┴────────┴────────┘
```

- **x-axis** uses colors 0,1 with entry-point task IDs 14,15 (the convention
  from `gemv-collectives_2d`). Each row independently gathers `K` u32s of
  distance + `K` u32s of index from each of its P PEs into the column-0 PE.
- **y-axis** uses colors 4,5 / task IDs 16,17. Each column gathers similarly
  into row 0.

Every PE participates in both gathers regardless of column. Cols 1..P-1 do a
"useless" y-gather whose output we never read; this avoids per-column
branching that would deadlock the SPMD collective and costs nothing extra
since those gathers run on disjoint y-fabric routes in parallel with col 0's.

Wavelets carry raw u32 payload — for distances we bit-cast `f32 ↔ u32`.

## 2. Local top-K algorithm

**Distance compute** (`compute_distances`): the loop nest is `(j, r)` instead
of `(r, j)` so we vectorize over `rows_per_pe` lanes via DSDs. Three DSD ops
per dimension:

```
for j in 0..d_dim:
  scratch[r] = D[r,j] - q[j]              fsubs DSD - scalar
  scratch[r] = scratch[r] * scratch[r]    fmuls DSD * DSD
  dist[r]   += scratch[r]                 fadds DSD + DSD
```

Cost ≈ `3 · d_dim` DSD-vectorized ops, dominated by FPU throughput. For
`k_large` (d=16, RPE=256) that's 48 DSD ops, ~12k FPU element-ops per PE.

**Local top-K extraction** (`topk_select`): max-heap of size K + heapsort.
Heapify O(K), each remaining element O(log K) (skip-fast on a tightening
threshold), heapsort O(K log K). Total **O((RPE − K) · log K + K · log K)**
vs. O(RPE · K) for selection sort. On `k_large` (K=RPE=256) this is roughly
a 32× win. Heapsort output is ascending in lex order — a property we exploit
in the merge stage.

**Merge stages** (`merge_sorted_runs`): each contributor sends its already-
ascending K-element block, so the row-merge input is `P` concatenated
sorted runs of length `K`. We do a P-way linear-min merge: `K` pops, each
scans the `P` heads and advances one. **O(K · P)** vs. O((K·P) · log K) for
re-heaping the K·P input. Gains scale with K — for `k_large` (K=256, P=2)
this is ~24× fewer compares per merge stage; the row-merge alone drops
from ~6k compares to ~512.

## 3. Fabric bandwidth accounting

Per gather: each PE sends `K` u32s into the row/column. Per axis we run two
gathers (distance + index), so 2K u32s per PE per axis.

Worst case `k_large` (K=256, P=2):
- x-fabric edge between adjacent cols carries up to P·K = 512 u32s per
  gather, 2 gathers → 1024 u32s.
- y-fabric edge between adjacent rows carries up to P·K = 512 u32s per
  gather, 2 gathers → 1024 u32s.

Worst case `baseline` (K=16, P=4): per-edge ≤ 4·16·2 = 128 u32s per axis.

The bottleneck is **compute, not routing.** Distance compute alone is
~12k FPU ops/PE on `k_large`; fabric carries < 1.1k u32s in any direction.
At ~1 cycle/wavelet on a free fabric, routing is comfortably masked by
the local heap and merge work.

## 4. Tie-break determinism

Every comparison uses `lex_lt(a_d, a_i, b_d, b_i) ≡ (a_d < b_d) ∨
(a_d == b_d ∧ a_i < b_i)` — a strict total order on `(distance, original_index)`
pairs that matches the oracle's lexsort.

- **Local stage**: `topk_select` only ever compares pairs via `lex_lt`, so
  its output is the K minima in that order regardless of the order entries
  appeared in `dist_buf`.
- **Merge stages**: `mpi_x.gather` lays contributions into `row_gather_*`
  in PE-id order (deterministic). The callback fires only after every
  contributor has landed, so the merge runs on a fixed buffer state — wavelet
  arrival order on the fabric is irrelevant.
- **Cross-PE ties**: each PE owns a contiguous global-index range
  (`index_offset + r`), so `(dist, idx)` uniquely identifies a row across
  the whole grid — no two distinct rows can produce the same key.
- **Composition**: `lex_lt` defines a total order, so "take the K minima"
  is associative under merge. Local-min → row-min → global-min yields the
  global K-minima, matching `topk_reference`'s `np.lexsort`.

The `all_equal` and `duplicates` cases exercise the second clause of
`lex_lt` directly; they pass because index is part of every comparison from
the very first row.

## 5. With 2× more time

1. **Pack dist + idx into a single gather.** Today we issue two K-wide
   gathers per axis. Interleaving them as one 2K-wide gather (dist bit-cast
   to u32) saves one gather-setup latency per axis — modest (~50–100 cycles
   per axis) but free of correctness risk once the unpack stride is right.
2. **Streaming compute → heap fusion.** Materialising `dist_buf` then
   heaping is two passes over the per-row data. Folding the heap-replace
   check into a row-serial distance loop saves the prefetch pass and the
   sentinel rewrite, at the cost of giving up DSD vectorisation in the
   distance compute. Whether this nets out positive depends on the actual
   DSD-vs-scalar throughput on WSE-2; worth measuring with `csdb` before
   committing.
