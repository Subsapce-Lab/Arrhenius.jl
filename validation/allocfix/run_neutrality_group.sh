#!/usr/bin/env bash
set -euo pipefail
project=$1
out=$2
list=$3
log=$4
tag=$5
cd ~/work/nh3-parity
source ./env.sh >/dev/null 2>&1
mkdir -p "$out" /tmp/allocfix-neutral-done
feed() {
  while IFS=$'\t' read -r id spec; do
    [[ -n "$id" ]] || continue
    printf '%s\t%s\t1\t1\t%s\n' "$spec" "$out/$id.npz" "/tmp/allocfix-neutral-done/$tag-$id.done"
  done < "$list"
  printf 'QUIT\n'
}
feed | ~/work/bin/run_capped 3600 "$JL" --project="$project" h/nh3_native_server.jl > "$log" 2>&1
while IFS=$'\t' read -r id spec; do
  [[ -n "$id" ]] || continue
  [[ -f "/tmp/allocfix-neutral-done/$tag-$id.done" ]] || { echo "missing completion marker for $id" >&2; exit 1; }
  grep -qx DONE "/tmp/allocfix-neutral-done/$tag-$id.done" || { echo "failed completion marker for $id" >&2; exit 1; }
done < "$list"
