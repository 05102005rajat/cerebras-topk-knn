# Top-K k-NN — Cerebras Kernel Challenge submission

**Start time:** 2026-04-29 (logged per SPEC §7).

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
`cs_python run.py --name <build_dir> --case <case>`; we look for the
`PASS: <case>` marker.

## What's where

- **Distance compute** is row-vectorized via DSDs (`pe_program.csl`,
  `compute_distances`).
- **Local top-K** is max-heap + heapsort (`topk_select`).
- **Merge stages** exploit the fact that each contributor's K-element
  block is already ascending — a P-way linear-min merge over the
  K-runs gives the merged top-K in O(K·P), avoiding a full re-heap of
  the K·P input (`merge_sorted_runs`).
- **Tie-break** is enforced by a single `lex_lt` total order on
  `(dist, global_idx)` used in every comparison.
- **Padding** for uneven shards is rewritten in-kernel to
  `(INF, U32MAX)` so it always sorts to the back.

See `DESIGN.md` for the full architectural rationale.
