#!/usr/bin/env bash
# Run all 6 grader cases in sequence. Stops on first failure.
set -e

declare -A CASES=(
  ["baseline"]="P:4,d_dim:32,rows_per_pe:128,K:16"
  ["k_eq_1"]="P:2,d_dim:32,rows_per_pe:256,K:1"
  ["k_large"]="P:2,d_dim:16,rows_per_pe:256,K:256"
  ["uneven"]="P:4,d_dim:32,rows_per_pe:64,K:16"
  ["all_equal"]="P:2,d_dim:16,rows_per_pe:256,K:16"
  ["duplicates"]="P:2,d_dim:16,rows_per_pe:256,K:8"
)

for case in baseline k_eq_1 k_large uneven all_equal duplicates; do
  params="${CASES[$case]}"
  out="out_${case}"
  echo "=== Compiling $case ($params) ==="
  cslc --arch=wse2 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
    --params="$params" --memcpy --channels=1 -o "$out"
  echo "=== Running $case ==="
  cs_python run.py --name "$out" --case "$case"
done
echo "ALL CASES PASSED"
