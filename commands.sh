#!/usr/bin/env bash
# Compile + run the baseline case end-to-end. Mirrors the grader's invocation
# in tests/test_correctness.py so a green run here implies a green grader.
set -euo pipefail

cslc --arch=wse2 ./layout.csl \
  --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 \
  --memcpy --channels=1 \
  -o out

cs_python run.py --name out --case baseline
