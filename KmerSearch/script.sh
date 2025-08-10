#!/usr/bin/env bash
set -euo pipefail

# Run ./main for every combination of ref_* and reads_*-*.txt with k in {10,25}.
# Output: out_<refN>_<readsN>_<k>
# e.g., ref_10000.txt + reads_100-50.txt + k=10 -> out_10000_100-50_10

DIR="${DIR:-.}"
cd "$DIR"
shopt -s nullglob

for ref in ref_*.txt; do
  ref_base="$(basename "$ref")"
  if [[ "$ref_base" =~ ^ref_([0-9]+)\.txt$ ]]; then
    refN="${BASH_REMATCH[1]}"
  else
    continue
  fi

  for reads in reads_*-*.txt; do
    reads_base="$(basename "$reads")"
    if [[ "$reads_base" =~ ^reads_([0-9]+-[0-9]+)\.txt$ ]]; then
      readsN="${BASH_REMATCH[1]}"
    else
      continue
    fi

    for k in 5 10; do
      out="out_${refN}_${readsN}_${k}.txt"
      echo ">>> Running: ./main \"$ref\" \"$reads\" $k \"$out\""
      ./main "$ref" "$reads" "$k" "$out"
    done
  done
done

echo "All runs completed."
