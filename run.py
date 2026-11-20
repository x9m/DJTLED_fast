#!/usr/bin/env python3
"""Choose a model, a solver and what to export, then run it.

    python3 run.py                                             pick from menus
    python3 run.py -m NH_001_tension -a DJTLED_fast            no questions asked (exports animation + final state)
    python3 run.py -m NH_001_tension -a all -e final           every solver, final state only
    python3 run.py -m NH_001_tension -a all -e none -t 4       same 4 threads for every solver, timing only
    python3 run.py --benchmark -m NH_001_tension -t 1,4,10     speed comparison: every solver, each 3 times per thread count

Nothing here is tied to particular names; the menus list what is in the folders (see discover.py):
Models  are the .txt files in models/.
Solvers are the .cpp files in src/ that have a main(); a solver's name is its file name without .cpp. For -a give that
        name, any part of it that fits only one solver (e.g. fast), or "all".
Export: "both" (animation + final state), "animation", "final" or "none".
Benchmark (--benchmark): runs the solvers without any export, --repeats times (default 3) for every thread count in
-t (a list such as 1,4,10 is allowed here), and saves the times in results/timings.csv. compare.py shows them, with the
speed-up, in its report.

Every solver is run under the same conditions, so that their times can be compared:
  threads      the same count for every solver: -t, or without it the number of performance cores (all cores on
               other machines)
  wait policy  the same OpenMP wait policy for every solver: --wait active (default; idle threads spin, best on an
               idle machine) or --wait passive (idle threads sleep; use it when the machine is busy with other work).
               --wait default sets no policy at all: each solver then runs as distributed, which is not the same for
               every solver (DJTLED gets the OpenMP runtime's default, DJTLED_fast asks for active waiting itself).
               A benchmark records the policy each solver reports it ran with, and no speed-up is given between
               solvers whose policies differ
  times        each solver reports its computation time (the time loop only) and, separately, its preparation time
               (reading the model and set-up); a benchmark records both, pauses between runs (--pause, default 2 s)
               and warns when the machine is busy
A solver is compiled here, into build/, when its executable is missing or older than its sources. The solvers write
their output files to results/ in this folder, named <solver>_<model>_...
A benchmark of a step count that was timed before (same model, solver and thread count) replaces those timings; a
partial run (--steps) and a full run are different step counts and are kept side by side.
"""
import argparse
import csv
import datetime
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True   # no __pycache__ folder: its files would hold this computer's folder path
import discover
from discover import ROOT, RESULTS

TIMINGS = RESULTS / "timings.csv"   # written by --benchmark, read by compare.py
TIMING_COLUMNS = ["model", "algorithm", "threads", "repeat", "computation_ms", "preparation_ms", "wait", "steps", "nodes", "elements", "fingerprint", "date"]
EXPORTS = {  # value passed to the solver's --export option: (menu text, endings of the files it writes)
    "both": ("animation and final state", ["_animation.xdmf", "_timesteps", "_deformed.vtk", "_undeformed.vtk"]),
    "animation": ("animation only", ["_animation.xdmf", "_timesteps"]),
    "final": ("final state only (deformed + undeformed)", ["_deformed.vtk", "_undeformed.vtk"]),
    "none": ("nothing, just run and time the solver", []),
}


def size_text(path):
    size = path.stat().st_size
    return f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} kB"


def ask(title, options):
    """Print a numbered menu and return the index of the chosen option."""
    print(f"\n{title}")
    for i, text in enumerate(options, 1):
        print(f"  {i}) {text}")
    while True:
        answer = input(f"Enter a number (1-{len(options)}): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print("  Please type one of the numbers above.")


def speed_text(ratio):
    """How a solver compares with the baseline, given baseline time / its time: '  (2.4x faster)', '  (1.3x slower)'."""
    if 0.95 <= ratio < 1.05:
        return "  (same speed)"
    return f"  ({ratio:.1f}x faster)" if ratio > 1 else f"  ({1 / ratio:.1f}x slower)"


def parse_threads(text):
    """'4' -> [4]; '1,4,10' -> [1, 4, 10]. Returns None if the text is not a list of whole numbers >= 1."""
    parts = [part.strip() for part in text.split(",")]
    if not parts or not all(part.isdigit() and int(part) >= 1 for part in parts):
        return None
    return [int(part) for part in parts]


def ask_threads(several):
    """Ask for the thread count(s), the same for every solver; an empty answer means the default count."""
    what = "a number or a list such as 1,4,10" if several else "a number"
    default = discover.default_threads()
    while True:
        answer = input(f"\nHow many threads for every solver? ({what}, or just Enter for {default}): ").strip()
        if answer == "":
            return [default]
        threads = parse_threads(answer)
        if threads and (several or len(threads) == 1):
            return threads
        print(f"  Please type {what} (whole numbers of 1 or more), or press Enter.")


def find_model(name, models):
    """Accept NH_001_tension, NH_001_tension.txt or a path to a model file."""
    for m in models:
        if name in (m.name, m.stem):
            return m
    path = Path(name).expanduser()
    if path.is_file():
        return path.resolve()
    sys.exit(f"Unknown model '{name}'. Available: {', '.join(m.stem for m in models) or 'none'}")


def pick_solvers(name, solvers):
    """'all' -> every solver; otherwise the solver with that name, or the only one whose name contains it."""
    if name.lower() == "all":
        return solvers
    exact = [s for s in solvers if s.name.lower() == name.lower()]
    partial = [s for s in solvers if name.lower() in s.name.lower()]
    if exact or len(partial) == 1:
        return exact or partial
    problem = "fits more than one solver" if partial else "is not a solver"
    sys.exit(f"'{name}' {problem}. Available: {', '.join(s.name for s in solvers)}, or all")


def openmp_flags():
    """Compiler and linker flags for OpenMP. macOS: Apple clang needs Homebrew's libomp."""
    if platform.system() != "Darwin":
        return ["-fopenmp"], ["-fopenmp"]
    libomp = Path("/opt/homebrew/opt/libomp")
    if shutil.which("brew"):
        found = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True).stdout.strip()
        libomp = Path(found) if found else libomp
    if not (libomp / "include" / "omp.h").is_file():
        sys.exit("OpenMP was not found. Install it with:  brew install libomp")
    return (["-Xpreprocessor", "-fopenmp", f"-I{libomp / 'include'}"],
            [f"-L{libomp / 'lib'}", "-lomp", f"-Wl,-rpath,{libomp / 'lib'}"])


def build(solvers):
    """Compile each chosen solver whose executable is missing or older than one of its sources."""
    shared, headers = discover.shared_sources(), discover.headers()
    flags = None   # looked up once, and only if something has to be compiled
    for solver in solvers:
        newest = max(f.stat().st_mtime for f in [solver.source, *shared, *headers])
        if solver.exe.is_file() and solver.exe.stat().st_mtime >= newest:
            continue
        flags = flags or openmp_flags()
        compile_flags, link_flags = flags
        command = ["c++", "-std=c++17", "-O3", f'-DSOLVER_NAME="{solver.name}"',   # the name in front of its output files
                   *compile_flags, str(solver.source), *map(str, shared), "-o", str(solver.exe), *link_flags]
        print(f"Compiling {solver.name} ...", flush=True)
        solver.exe.parent.mkdir(exist_ok=True)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"Compiling {solver.source.name} failed:\n{' '.join(command)}\n{result.stderr}")


def solver_environment(wait):
    """The environment every solver is started with: the same OpenMP wait policy for all of them.
    "default" leaves the policy unset (not the same as "passive"), also when the calling shell has set one."""
    if wait == "default":
        return {name: value for name, value in os.environ.items() if name != "OMP_WAIT_POLICY"}
    return dict(os.environ, OMP_WAIT_POLICY=wait)


def written_since(path, start):
    """True if the file, or for a folder any file in it, was written at or after `start` (seconds since the epoch).
    A time without a fraction comes from a file system that keeps whole seconds only, and is compared that way."""
    def fresh(f):
        written = f.stat().st_mtime
        return written >= (math.floor(start) if written == int(written) else start)
    if path.is_dir():
        return any(fresh(f) for f in path.iterdir())
    return path.is_file() and fresh(path)


def run(solver, model, export, threads, wait, steps=0):
    command = [str(solver.exe), str(model), "--export", export, "--threads", str(threads)]
    if steps:
        command += ["--steps", str(steps)]
    print(f"\n=== {solver.name} on {model.name} (export: {export}, threads: {threads}, wait policy: {wait}) ===", flush=True)
    started = time.time()
    code = subprocess.run(command, cwd=ROOT, env=solver_environment(wait)).returncode  # the solver creates results/ where it runs
    # A run cut short by --steps writes under <solver>_stepsN_<model>. With N at or above the model's step count the
    # solver runs in full and keeps the usual name, so both names are looked at and only what this run wrote is listed.
    prefixes = ([f"{solver.name}_steps{steps}_{model.stem}"] if steps else []) + [f"{solver.name}_{model.stem}"]
    if code == 0:
        written = [prefix + e for prefix in prefixes for e in EXPORTS[export][1] if written_since(RESULTS / (prefix + e), started)]
        print(f"\n{solver.name} finished." + (f" Output in {RESULTS}:" if written else " No files were written."))
        for name in written:
            print(f"  {name}")
    else:
        print(f"\n{solver.name} FAILED (exit code {code}); see the messages above.")
    return code


def benchmark(solvers, model, thread_list, repeats, wait="active", pause=2.0, steps=0):
    """Time the chosen solvers without any export, all under the same conditions, and save the times in results/timings.csv."""
    grab = lambda pattern, text: (re.search(pattern, text) or [None, None])[1]
    width = max(len(s.name) for s in solvers)
    # identity of what is being measured, so compare.py can discard timings taken before an edit
    shared, headers = discover.shared_sources(), discover.headers()
    fingerprint = {s.name: discover.file_fingerprint(model, s.source, *shared, *headers) for s in solvers}
    rows, failed = [], False
    print(f"\nBenchmark on {model.name}: {repeats} run(s) per solver and thread count, no export, wait policy {wait}.")
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    if load > 2.0:
        print(f"  WARNING: the machine is busy (load {load:.1f}); these timings will be slower and less repeatable than on an idle machine"
              + (" - consider --wait passive." if wait == "active" else "."))
    first_run = True
    for threads in thread_list:
        for repeat in range(1, repeats + 1):
            for solver in solvers:   # the solvers take turns, so a machine that warms up slows all alike
                command = [str(solver.exe), str(model), "--export", "none", "--threads", str(threads)]
                if steps:
                    command += ["--steps", str(steps)]
                if not first_run:
                    time.sleep(pause)   # let the machine settle between runs (nothing to wait for after the last one)
                first_run = False
                done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=solver_environment(wait))
                ms, used = grab(r"Computation time:\s*(\d+) ms", done.stdout), grab(r"using (\d+) threads", done.stdout)
                # the policy the solver really ran with: a solver may choose one itself when none is given (--wait default)
                effective = grab(r"wait policy: (\w+)", done.stdout) or wait
                if done.returncode != 0 or ms is None or used is None:
                    failed = True
                    message = (done.stderr.strip().splitlines() or ["no computation time in the output"])[-1].strip()
                    print(f"  {solver.name:<{width}} {threads:>7} threads  run {repeat}/{repeats}:  FAILED ({message})")
                    continue
                print(f"  {solver.name:<{width}} {used:>7} threads  run {repeat}/{repeats}: {int(ms):>9,} ms", flush=True)
                rows.append({"model": model.stem, "algorithm": solver.name, "threads": int(used), "repeat": repeat, "computation_ms": int(ms),
                             "preparation_ms": grab(r"Preparation time:\s*(\d+) ms", done.stdout), "wait": effective,
                             "steps": grab(r"NumSteps:\s*(\d+)", done.stdout), "nodes": grab(r"Nodes:\s*(\d+)", done.stdout),
                             "elements": grab(r"Elements:\s*(\d+)", done.stdout),
                             "fingerprint": fingerprint[solver.name],
                             "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
    if rows:
        # keep earlier timings of other models / solvers / thread counts / step counts; replace the ones measured again
        # now. The step count is part of the key, so a partial run (--steps) and a full run are kept side by side.
        measured = {(r["model"], r["algorithm"], str(r["threads"]), str(r["steps"] or "")) for r in rows}
        kept = []
        if TIMINGS.is_file():
            with open(TIMINGS, newline="") as f:
                kept = [r for r in csv.DictReader(f)
                        if (r.get("model"), r.get("algorithm"), r.get("threads"), r.get("steps") or "") not in measured]
        RESULTS.mkdir(exist_ok=True)
        with open(TIMINGS, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TIMING_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept + rows)
        # median run per solver; the speed-up is against the first solver (the baseline, see discover.py)
        column = max(24, width + 20)
        print("\n  threads" + "".join(f"{s.name + ', median':>{column}}" for s in solvers))
        unlike = {}   # solver -> wait policies, where the solvers of a row did not run with the same one
        for threads in sorted({r["threads"] for r in rows}):
            med, waits = [], []
            for s in solvers:
                mine = [r for r in rows if r["threads"] == threads and r["algorithm"] == s.name]
                med.append(statistics.median(r["computation_ms"] for r in mine) if mine else None)
                waits.append({r["wait"] for r in mine})
            line = f"  {threads:>7}"
            for k, b in enumerate(med):
                text = "-" if b is None else f"{b:,.0f} ms"
                if k > 0 and b and med[0]:
                    if waits[k] == waits[0]:
                        text += speed_text(med[0] / b)
                    else:   # a ratio of runs made under unlike conditions would mislead
                        text += "  (*)"
                        for i in (0, k):
                            unlike.setdefault(solvers[i].name, set()).update(waits[i])
                line += f"{text:>{column}}"
            print(line)
        if unlike:
            print("  (*) no speed-up shown: the solvers ran with different wait policies ("
                  + "; ".join(f"{name}: {', '.join(sorted(policies))}" for name, policies in unlike.items()) + ").")
        print(f"\nTimings saved in {TIMINGS}. Run  python3 compare.py  to see them in the HTML report.")
    return 1 if failed or not rows else 0


def main():
    parser = argparse.ArgumentParser(description="Choose a DJ-TLED model, solver and export, then run it.")
    parser.add_argument("-m", "--model", help="model name in models/ (with or without .txt) or a path to a model file")
    parser.add_argument("-a", "--algorithm", help="solver name (a .cpp file in src/ with a main()), a part of it that fits "
                                                  "only one solver, or 'all'")
    parser.add_argument("-e", "--export", choices=list(EXPORTS), help="output files to write (default with -m and -a: both)")
    parser.add_argument("-t", "--threads", help="threads for every solver run (default: the performance cores, the same for every solver); "
                                                "with --benchmark a list such as 1,4,10")
    parser.add_argument("--steps", type=int, default=0, help="stop every solver after N steps (a partial run); 0 means the full run")
    parser.add_argument("-b", "--benchmark", action="store_true", help="time the solvers (no export) and save results/timings.csv")
    parser.add_argument("--repeats", type=int, default=3, help="--benchmark: runs per solver and thread count (default 3)")
    parser.add_argument("--pause", type=float, default=2.0, help="--benchmark: seconds to wait between runs (default 2)")
    parser.add_argument("--wait", choices=["active", "passive", "default"], default="active",
                        help="OpenMP wait policy given to every solver: active (default), passive (for a busy machine), "
                             "or default (set no policy: each solver runs as distributed, the paper's 'default wait' "
                             "column for DJTLED; DJTLED_fast then still asks for active waiting itself)")
    args = parser.parse_args()
    thread_list = None
    if args.threads is not None:
        thread_list = parse_threads(args.threads)
        if thread_list is None:
            parser.error("--threads must be a whole number of 1 or more" + (" or a list such as 1,4,10" if args.benchmark else ""))
        if len(thread_list) > 1 and not args.benchmark:
            parser.error("--threads takes one number (a list is only for --benchmark)")
    if args.repeats < 1:
        parser.error("--repeats must be 1 or more")
    if args.pause < 0 or not math.isfinite(args.pause):
        parser.error("--pause must be a finite number of 0 or more")
    if args.steps < 0:
        parser.error("--steps must be 0 (full run) or a whole number of 1 or more")

    all_solvers = discover.find_solvers()
    if not all_solvers:
        sys.exit(f"No solver found: {discover.SOURCES} has no .cpp file with a main().")
    models = discover.find_models()
    if args.model:
        model = find_model(args.model, models)
    else:
        if not models:
            sys.exit(f"No .txt model files found in {discover.MODELS}")
        model = models[ask("Which model?", [f"{m.stem}  ({size_text(m)})" for m in models])]

    if args.algorithm:
        solvers = pick_solvers(args.algorithm, all_solvers)
    elif args.benchmark or len(all_solvers) == 1:
        solvers = all_solvers
    else:
        labels = [f"{s.name}  (src/{s.source.name})" for s in all_solvers] + ["all of them, one after the other"]
        pick = ask("Which algorithm?", labels)
        solvers = all_solvers if pick == len(all_solvers) else [all_solvers[pick]]

    if args.benchmark:
        if thread_list is None:
            thread_list = [discover.default_threads()] if args.model else ask_threads(several=True)
        build(solvers)
        sys.exit(benchmark(solvers, model, thread_list, args.repeats, args.wait, args.pause, args.steps))

    if args.export:
        export = args.export
    elif args.model and args.algorithm:
        export = "both"
    else:
        export = list(EXPORTS)[ask("What to export?", [text for text, _ in EXPORTS.values()])]

    if thread_list is None:
        thread_list = [discover.default_threads()] if (args.model and args.algorithm) else ask_threads(several=False)
    threads = thread_list[0]   # the same count for every solver

    build(solvers)
    codes = [run(solver, model, export, threads, args.wait, args.steps) for solver in solvers]
    sys.exit(1 if any(codes) else 0)  # a solver killed by a signal has a negative code


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nCancelled.")
