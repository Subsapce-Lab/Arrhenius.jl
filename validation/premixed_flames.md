# Premixed-flame validation

The complete free, burner and prescribed-temperature examples were checked with
Julia 1.12.7 and Cantera 4.0.0a2 at revision
[726522be](https://github.com/Cantera/cantera/tree/726522be4e2a13454d8415b7ef799d621f665cf3).
[Results and all timing samples](results/cantera4_flames_5ca6641.json) identify
native revision 5ca6641 and the source, mechanism and reference hashes.

| Complete source calculation | WSL speed ratio | Apple M4 speed ratio | Both hosts ≥0.95 |
| --- | ---: | ---: | --- |
| Free flame, four transport stages | 0.738 | 0.660 | No |
| Burner flame, two transport stages | 2.240 | 2.159 | Yes |
| Prescribed-temperature flame, two transport stages | 1.476 | 1.932 | Yes |

Ratios are median Cantera time divided by median Julia time, with nine warm
complete sequences and one numerical thread per implementation. Each repetition's
stage times are summed before taking the median. Timers include flame construction,
initialization, continuation and adaptive solves; mechanism/sidecar preparation,
profile snapshots and file output are outside them. The first measured sequence is
reported separately. Its internal stage timers exclude startup/imports and some
specialization, so those first times are not whole-process cold latency.

Every transport stage passes the existing full-profile and elemental-conservation
checks against independently refined Cantera references. The acceptance limits are
1% for temperature profiles and free-flame speed, 5% for species profiles with a
1e-7 reference-peak floor, and 1e-6 for elemental conservation. Numerical replay
checks all first and warm results at relative tolerance 1e-12 and absolute tolerance
1e-14. Accuracy-reference refinement is outside every timed calculation.

The [shared Julia calculation](../example/flames/source_flame_sequence.jl) is used
by the runnable examples and the [paired benchmark](conservative_source_bench.py).
Run the benchmark with `--help` for its required provenance and reference inputs.
These results cover the listed source conditions and transport sequences; the free
flame still needs performance improvement.
