#!/usr/bin/env bash
set -euo pipefail
base_project=$1
candidate_project=$2
out_root=$3
cases=$4
source ~/work/nh3-parity/env.sh
cd ~/work/nh3-parity
run_version() {
  local project=$1 version=$2 out="$out_root/$2"
  mkdir -p "$out" /tmp/allocfix-timing-done
  feed() {
    while IFS=$'\t' read -r id spec; do
      [[ -n "$id" ]] || continue
      printf '%s\t%s\t3\t1\t%s\n' "$spec" "$out/$id.npz" "/tmp/allocfix-timing-done/$version-$id.done"
    done < "$cases"
    printf 'QUIT\n'
  }
  feed | ~/work/bin/run_capped 3600 "$JL" --project="$project" h/nh3_native_server.jl > "$out/server.log" 2>&1
  while IFS=$'\t' read -r id spec; do
    [[ -n "$id" ]] || continue
    [[ -f "/tmp/allocfix-timing-done/$version-$id.done" ]] || { echo "missing completion marker for $id ($version)" >&2; return 1; }
    grep -qx DONE "/tmp/allocfix-timing-done/$version-$id.done" || { echo "failed completion marker for $id ($version)" >&2; return 1; }
  done < "$cases"
}
run_version "$base_project" main
run_version "$candidate_project" candidate
