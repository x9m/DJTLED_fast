# An improvement on direct Jacobian total Lagrangian explicit dynamics (DJ-TLED)

A faster implementation of the DJ-TLED finite element solver for soft-tissue simulation, together with the original
solver, two test models, and scripts to run and compare them.

DJ-TLED is the algorithm described in:

> Zhang, J. (2021). A direct Jacobian total Lagrangian explicit dynamics finite element algorithm for real-time
> simulation of hyperelastic materials. *International Journal for Numerical Methods in Engineering*, 122(20),
> 5744–5772. doi:10.1002/nme.6772

Both solvers here use 4-node tetrahedral elements, a neo-Hookean material and explicit central-difference time
integration, in single precision, parallelised with OpenMP.

| Solver | Source | What it is |
|---|---|---|
| `DJTLED` | `src/DJTLED.cpp` | Zhang's original code, edited only so both solvers share the same validation, boundary-condition rules, timing and output (the element kernel is unchanged) |
| `DJTLED_fast` | `src/DJTLED_fast.cpp` | The same algorithm and input format, reorganised for speed |

## Results

Measured on an Apple-silicon MacBook Air (M5) with 4 performance and 6 efficiency cores. Times are for the time loop
only, with no file reading, set-up or VTK export. Both solvers were given the same thread count, and `DJTLED` is shown
with whichever OpenMP wait policy (active, or the runtime's default as distributed) was faster for it.

**Speed, small model** (`NH_001_tension`: 6,713 elements, complete run of 4,000 steps; median of 5 runs):

| Threads | DJTLED | DJTLED_fast | Speed-up |
|---|---|---|---|
| 1 | 434 ms | 203 ms | 2.1× |
| 2 | 286 ms | 123 ms | 2.3× |
| 4 | 186 ms | 78 ms | 2.4× |
| 6 | 275 ms | 105 ms | 2.6× |
| 10 | 243 ms | 89 ms | 2.7× |

**Speed, large model** (`NH_0002_tension`: 361,175 elements, time step 0.000025; first 2,001 steps, median of 2 runs):

| Threads | DJTLED | DJTLED_fast | Speed-up |
|---|---|---|---|
| 1 | 13.4 s | 5.5 s | 2.5× |
| 2 | 20.9 s | 3.0 s | 7.1× |
| 4 | 13.8 s | 2.1 s | 6.7× |
| 6 | 15.5 s | 2.5 s | 6.2× |
| 10 | 13.0 s | 2.1 s | 6.1× |

A complete run of the large model (40,000 steps, 4 threads) took 325.6 s with `DJTLED`. `DJTLED_fast` took 62.9 s
when run straight afterwards on the warm machine, and 45.4 s in a separate run from a cool machine: 5.2× to 7.2×.

**Accuracy** (largest nodal difference, as a percentage of the 0.05 prescribed displacement):

| Model | DJTLED vs reference | DJTLED_fast vs reference | DJTLED_fast vs DJTLED |
|---|---|---|---|
| `NH_001_tension` | 0.0015 % | 0.0018 % | 0.0017 % |
| `NH_0002_tension` | 0.051 % | 0.043 % | 0.042 % |

These numbers come from one machine that slows down when it gets warm, so treat the timings as indicative. The
small-model times include a wake-up cost of about 8 to 25 ms from the cooling pause between runs; the speed-ups are
therefore slightly conservative (about 2.4–3.1× without that pause). You can
reproduce them with the commands below. The full study, with the method behind every change, is in
[A Faster DJ-TLED (paper).pdf](<A Faster DJ-TLED (paper).pdf>).

## Requirements

- macOS. Everything here was developed and tested on macOS (Apple silicon).
- A C++17 compiler, available as `c++`.
- OpenMP. On macOS: `brew install libomp`.
- Python 3. The scripts use only the standard library.
- [ParaView](https://www.paraview.org) to look at the results (optional).

## Quick start

```
python3 run.py                # pick a model, a solver, what to export and the thread count from menus
python3 compare.py            # compare the results and open an HTML report
```

`run.py` compiles a solver the first time it is needed and again whenever its source changes. There is no separate
build step.

## Running a solver: `run.py`

```
python3 run.py -m NH_001_tension -a DJTLED_fast             # one solver; writes the animation and the final state
python3 run.py -m NH_001_tension -a all -e final            # every solver, final state only
python3 run.py -m NH_001_tension -a all -e none -t 4        # timing only, 4 threads each
python3 run.py --benchmark -m NH_001_tension -t 1,4,10      # speed comparison, 3 runs per solver and thread count
```

| Option | Meaning |
|---|---|
| `-m`, `--model` | A model name from `models/` (with or without `.txt`), or a path to a model file |
| `-a`, `--algorithm` | A solver name, a part of it that fits only one solver (for example `fast`), or `all` |
| `-e`, `--export` | `both` (animation and final state), `animation`, `final` or `none` |
| `-t`, `--threads` | Number of threads, the same for every solver. Without it: the number of performance cores |
| `--wait` | OpenMP wait policy, the same for every solver: `active` (default) or `passive` (for a busy machine). `default` sets no policy, so each solver runs as distributed: `DJTLED` with the runtime's default, `DJTLED_fast` still with active waiting. The timings record the policy each solver reports, and no speed-up is shown between solvers whose policies differ |
| `--steps` | Stop every solver after N steps (a partial run, see below); also works with `--benchmark`, where partial and full runs are kept as separate timings |
| `-b`, `--benchmark` | Time the solvers without any export and save the times in `results/timings.csv` |
| `--repeats` | Runs per solver and thread count in a benchmark (default 3) |
| `--pause` | Seconds to wait between benchmark runs so the machine can cool down (default 2) |

**Fair timing.** `run.py` always gives every solver the same thread count and the same wait policy, so their times
can be compared. Each solver prints two times: the *computation time* (the time loop only, which is what is compared)
and the *preparation time* (reading the model and set-up). A benchmark alternates the solvers, pauses between runs,
and warns when the machine is busy. Close other programs first: with the active wait policy, a busy machine can make
a run many times slower.

The solvers can also be run directly once they are built:

```
./build/djtled_fast models/NH_001_tension.txt --export final --threads 4
./build/djtled      models/NH_001_tension.txt --export none
```

Run directly, each solver uses its own defaults (`djtled`: all cores and the OpenMP runtime's default waiting;
`djtled_fast`: performance cores and active waiting). To compare times that way, give both the same `--threads` and set `OMP_WAIT_POLICY=active`
for `djtled`.

`djtled_fast` also accepts `--no-reorder`, which switches off its node and element reordering.

Both solvers accept `--steps N`, which stops after N steps (N up to the full step count) while keeping the loading
ramp over the full total time. It reproduces a partial run, such as the first 2,001 steps of the large model:
`./build/djtled_fast models/NH_0002_tension.txt --export none --threads 4 --steps 2001`. When `--steps` caps the run,
any output files are named `<solver>_stepsN_<model>_…` so a partial run does not overwrite the full one.

## Comparing results: `compare.py`

You pick one baseline, either a solver result or a reference file, and what to compare with it:

```
python3 compare.py --one DJTLED_NH_001_tension --with DJTLED_fast_NH_001_tension                          # one other
python3 compare.py --one DJTLED_NH_001_tension --with DJTLED_fast_NH_001_tension U_DJ_Reference_ten_001   # several
python3 compare.py --one DJTLED_NH_001_tension --with all                                                 # all others
```

"All" means everything with the same number of points as the baseline. Every selected item is compared in two ways:

- **Errors.** For each point, the difference from the baseline in x, y and z and its magnitude. The script reports
  the largest difference in each direction, the point id where it occurs, and the RMSE over the whole model. A
  comparison passes when its largest difference is at most `--tol` percent (default 0.5) of the largest displacement
  in the baseline. The exit code is 1 if any comparison fails.
- **Speed.** The computation times of the solvers behind the selected results, with the speed-up against the
  baseline's solver. A solver that has not been timed on that model yet (or whose timings are out of date) is timed
  after you confirm, using the menus; `--no-timing` skips that, and a non-interactive `--one … --with` run never
  re-times on its own. Timings of a full run and of a partial run (`--steps`) are shown as separate tables, and a
  row shows no speed-up when its solvers did not run with the same wait policy.

It writes to `results/`:

| File | Contents |
|---|---|
| `compare_report.html` | The report: summary, speed table and chart, error distribution, error maps. Opened in the browser |
| `compare_<other>_vs_<baseline>.csv` | The per-point table |
| `compare_<other>_vs_<baseline>.vtk` | The error map for ParaView; colour by `difference_magnitude` |

Other options: `--no-csv`, `--no-vtk` and `--no-open` skip the per-point tables, the error maps and opening the
browser.

## Output files

The solvers write to `results/`, named after the solver and the model:

| File | Contents |
|---|---|
| `<solver>_<model>_deformed.vtk` | Final state, deformed mesh with the displacement field |
| `<solver>_<model>_undeformed.vtk` | Final state on the undeformed mesh |
| `<solver>_<model>_animation.xdmf` | The animation; open this file in ParaView and choose the "XDMF Reader" |
| `<solver>_<model>_timesteps/` | The animation's data: the elements once (`topology.bin`), and the deformed coordinates and displacements of each frame (raw binary), about 100 frames per run |

The animation of the large model is about 170 MB per solver; use `-e final` to skip it. The elements do not change
during a run, so they are stored once and every frame refers to them (the earlier one-VTK-file-per-frame animation
repeated them in each frame and took about 1 GB). The `.xdmf` file refers to the data by relative path, so keep it
next to its `_timesteps/` folder. The data files are raw 32-bit values in the byte order of the machine that wrote
them, which the `.xdmf` file records.

## Models

| Model | Nodes | Elements | Time step | Steps |
|---|---|---|---|---|
| `NH_001_tension` | 1,440 | 6,713 | 0.00025 | 4,000 |
| `NH_0002_tension` | 67,362 | 361,175 | 0.000025 | 40,000 |

Both are a cube pulled by 0.05 in y on one face over a total time of 1, with a neo-Hookean material (μ = 6,567,
K = 326,210) and density 1,060. In both models the pulled face is also fixed in x and z, and the opposite face is
fixed in all directions. `reference/` holds a reference result for each. These two models and their references were
prepared for this study; they are not part of Zhang's upstream repository (which ships different examples), and the
exact build and step that produced each reference file is not recorded.

A model is a plain text file:

```
0   0.1   0.1   0.1        node lines: id x y z
...
NH 6567 326210             material: neo-Hookean, Mu and K
Density 1060
T4                         element type, then element lines: id n1 n2 n3 n4
0   602 603 604 605
...
<Disp>                     prescribed displacement: direction and magnitude, then node ids
y 0.05
0 3 6 7 ...
<FixP>                     fixed nodes: x, y, z or all, then node ids
all
12 13 14 ...
</BC>
Damping 5
TimeStep 0.00025
TotalTime 1
```

A `<Gravity>` block (direction and acceleration) is also accepted. Node and element ids may start at 0 or 1.

Two things to know when making or changing a model:

- **The time step has a stability limit.** If it is too large the run stops with "an element inverted" or "solution
  diverged". On `NH_0002_tension` the limit is between 0.00004 and 0.000045.
- **Results are single precision.** On the large model, changing the time step moves the final displacements by up
  to about 0.3 % of the prescribed displacement, with no smooth trend. A difference smaller than that between two
  runs is not evidence of an error in either solver.

## What makes `DJTLED_fast` faster

- **Compact element data in blocks of eight.** Each element stores only the 21 values the force calculation needs,
  and the element kernel processes eight elements at once with SIMD.
- **A cheaper J^(-2/3).** It is refined from the previous step's value with two Newton iterations instead of a
  call to `powf`, with an exact fallback when the previous value is too far off.
- **One division instead of ten** in the element force, by using the adjugate of the Jacobian and a stored
  reciprocal of the initial determinant.
- **One thread team for the whole run**, with spinning rather than sleeping between the two phases of a step.
- **Node and element reordering** (reverse Cuthill–McKee), so neighbouring data sits close together in memory.
- **One thread per performance core** by default on Apple silicon.

It also checks that no element is inverted, both in the input mesh and at every step.

Compile-time settings at the top of `src/DJTLED_fast.cpp` (`FAST_W`, `FAST_SPIN`, `FAST_PCORES`, `FAST_POWF`,
`FAST_PERSTEP`, `J23_TOL`, `J23_ITERS`) can be overridden with `-DNAME=value` when compiling by hand; `run.py`
always builds with the defaults. `FAST_POWF` and `FAST_PERSTEP` are the two paper ablations (`powf` instead of the
warm-started iteration, and one parallel region per step).

## Project layout

```
src/         DJTLED.cpp, DJTLED_fast.cpp, and vtk_export.* (VTK output shared by both solvers)
models/      the model files
reference/   reference results
run.py       build and run the solvers, benchmark
compare.py   compare results, HTML report
discover.py  finds models, solvers, results and references by looking in the folders
build/       compiled solvers (created by run.py, not in git)
results/     solver output, comparison files, timings (not in git)
A Faster DJ-TLED (paper).pdf / .tex   the paper comparing the two solvers, and its LaTeX source
DJTLED Python/   DJTLED.ipynb, a NumPy port of DJTLED.cpp as an explained notebook (needs numpy; see its requirements.txt)
```

Nothing in the scripts is tied to particular file names. Any `.txt` file in `models/` is a model, any `.cpp` file in
`src/` with a `main()` is a solver, and any `.vtk` file in `reference/` is a reference. To add a solver, put its
source in `src/`; it must accept `--export` and `--threads` (and `--steps`, if you use that option of `run.py`), write its files through `vtk_export`, and print the lines
the scripts read: `using N threads`, `Computation time: N ms`, `NumSteps: N` and `Elements: N` (and, to have them
reported, `Preparation time: N ms` and, after the thread count, `(wait policy: NAME)`).

## Licence

MIT; see [LICENSE](LICENSE). The original DJ-TLED code is copyright (c) 2022 Jinao Zhang. The modifications and
`DJTLED_fast` are copyright (c) 2026 Baiqi Cheng.
