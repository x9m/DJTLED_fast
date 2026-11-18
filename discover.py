"""Find the project's models, solvers, results and reference files by looking in the folders.

run.py and compare.py use this instead of fixed lists of names, so they keep working when a model, a solver source
or a reference file is renamed, added or removed. Only the folder names are fixed:

    models/      every .txt file is a model; its name is the file name without .txt
    src/         every .cpp file with a main() is a solver; its name is the file name without .cpp.
                 The other .cpp files (e.g. vtk_export.cpp) are shared code, compiled into every solver.
    build/       the compiled solvers (made by run.py)
    results/     what the solvers and compare.py write; a result is <solver>_<model>_deformed.vtk
    reference/   every .vtk file is a reference result

run.py compiles each solver with its own name (-DSOLVER_NAME="<name>"), and the solver puts that name in front of its
output files. That is how a result file can be traced back to its solver and model here.
"""
import os
import platform
import re
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SOURCES = ROOT / "src"
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
REFERENCES = ROOT / "reference"

Solver = namedtuple("Solver", "name source exe")   # name: also the start of its output file names


def baseline_first(names):
    """Order solver names so the baseline comes first: the shortest name, then alphabetical (DJTLED before DJTLED_fast)."""
    return sorted(names, key=lambda name: (len(name), name))


def _has_main(source):
    return re.search(r"\bint\s+main\s*\(", source.read_text(errors="replace")) is not None


def find_models():
    """The model files, smallest first."""
    return sorted(MODELS.glob("*.txt"), key=lambda p: p.stat().st_size)


def find_solvers():
    """The solvers: one per .cpp file in src/ that has a main(), baseline first."""
    by_name = {source.stem: source for source in SOURCES.glob("*.cpp") if _has_main(source)}
    return [Solver(name, by_name[name], BUILD / name.lower()) for name in baseline_first(by_name)]


def shared_sources():
    """The .cpp files in src/ without a main(): compiled into every solver."""
    return sorted(source for source in SOURCES.glob("*.cpp") if not _has_main(source))


def headers():
    return sorted([*SOURCES.glob("*.h"), *SOURCES.glob("*.hpp")])


def find_results():
    """The final-state files written by the solvers."""
    return sorted(RESULTS.glob("*_deformed.vtk"))


def find_references():
    return sorted(REFERENCES.glob("*.vtk"))


def split_result_name(name):
    """'DJTLED_fast_NH_001_tension' -> ('DJTLED_fast', 'NH_001_tension').

    The solver is recognised by the solver names found in src/, or else the model by the model names found in
    models/ (longest name first, so DJTLED_fast is not mistaken for DJTLED). Returns (None, name) if neither fits,
    for example a result of a solver and a model that have both been renamed since."""
    for solver in sorted((s.name for s in find_solvers()), key=len, reverse=True):
        if name.startswith(solver + "_") and len(name) > len(solver) + 1:
            return solver, name[len(solver) + 1:]
    for model in sorted((m.stem for m in find_models()), key=len, reverse=True):
        if name.endswith("_" + model) and len(name) > len(model) + 1:
            return name[:-len(model) - 1], model
    return None, name


def default_threads():
    """The thread count every solver gets when none is asked for, so that their times can be compared: the performance
    cores on Apple silicon (the efficiency cores are much slower and would hold every step back), otherwise all cores."""
    if platform.system() == "Darwin":
        try:
            import ctypes
            import ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            cores, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
            if libc.sysctlbyname(b"hw.perflevel0.physicalcpu", ctypes.byref(cores), ctypes.byref(size), None, 0) == 0 and cores.value > 0:
                return cores.value
        except (OSError, AttributeError):
            pass
    return os.cpu_count() or 1


_point_counts = {}


def point_count(vtk_file):
    """Number of points in a legacy VTK file (from its POINTS line, without reading the whole file); None if not found."""
    key = (str(vtk_file), vtk_file.stat().st_mtime)
    if key not in _point_counts:
        count = None
        with open(vtk_file, errors="replace") as f:
            for _, line in zip(range(20), f):
                if line.startswith("POINTS"):
                    count = int(line.split()[1])
                    break
        _point_counts[key] = count
    return _point_counts[key]
