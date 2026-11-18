#!/usr/bin/env python3
"""Compare results with one baseline, point by point and in speed, and build an HTML report.

You pick ONE baseline (a solver result or a reference file) and what to compare with it:

    one other         python3 compare.py --one DJTLED_NH_001_tension --with DJTLED_fast_NH_001_tension
    several others    python3 compare.py --one DJTLED_NH_001_tension --with DJTLED_fast_NH_001_tension U_DJ_Reference_ten_001
    all others        python3 compare.py --one DJTLED_NH_001_tension --with all
    from menus        python3 compare.py

"All others" is everything with the same number of points as the baseline: the other solvers' results for that model
and its reference file. Whichever of the three you choose, every selected item is compared with the baseline in two ways:

  errors  For every point id the difference (other - baseline) in x, y and z and its magnitude. Reported: the largest
          difference in x, y, z and magnitude (and the point id where it occurs) and the RMSE over the whole model.
          A comparison PASSES when the largest difference magnitude is at most --tol percent (default 0.5) of the
          largest displacement in the baseline; the exit code is 1 if any comparison fails.
  speed   The computation times of the solvers behind the selected results, from results/timings.csv, with the
          speed-up against the baseline's solver. A solver that has not been timed on that model yet is timed now
          (one run, no export, the same thread count as the others; the menus ask first, --no-timing skips it).
          python3 run.py --benchmark measures more carefully (3 runs, several thread counts). A reference file has no
          solver, so it has no speed.

Nothing here is tied to particular names; everything is found in the folders (see discover.py). Results are the
final-state files results/<solver>_<model>_deformed.vtk written by the solvers (python3 run.py); references are the
.vtk files in reference/. A name or a path to a .vtk file can be given for the baseline and for the others.

Written to results/:
    compare_report.html               the report: summary, speed, charts and error maps; opened in the browser
    compare_<other>_vs_<baseline>.csv    the per-point table
    compare_<other>_vs_<baseline>.vtk    the error map for ParaView (colour by difference_magnitude)
All lengths are in the length unit of the model. Point ids count from 0, in the order of the VTK file.
"""
import argparse
import csv
import datetime
import html
import json
import math
import statistics
import sys
import webbrowser
from pathlib import Path

sys.dont_write_bytecode = True   # no __pycache__ folder: its files would hold this computer's folder path
import discover
from discover import RESULTS, REFERENCES

TIMINGS = RESULTS / "timings.csv"
REPORT = RESULTS / "compare_report.html"
MAP_POINTS = 20000                        # at most this many points are drawn in the report's error maps


# ------------------------------------------------------------------------------------------------- reading files
def take_values(lines, start, count, path):
    """Read count floats from lines[start:] (any number per line). Returns (values, index of the next line)."""
    values, i = [], start
    while len(values) < count:
        if i >= len(lines):
            sys.exit(f"{path}: the file ends before all its values")
        values.extend(float(v) for v in lines[i].split())
        i += 1
    return values, i


def read_vtk(path):
    """Read a legacy ASCII VTK file: point coordinates, displacement vectors and the lines describing the mesh."""
    with open(path) as f:
        lines = f.read().splitlines()
    try:
        i_points = next(i for i, l in enumerate(lines) if l.startswith("POINTS"))
        i_data = next(i for i, l in enumerate(lines) if l.startswith("POINT_DATA"))
        i_vectors = next(i for i in range(i_data, len(lines)) if lines[i].startswith("VECTORS"))
    except StopIteration:
        sys.exit(f"{path}: not a VTK file with POINTS and POINT_DATA / VECTORS")
    n = int(lines[i_points].split()[1])
    xyz, _ = take_values(lines, i_points + 1, 3 * n, path)
    u, _ = take_values(lines, i_vectors + 1, 3 * n, path)
    return {"points": [tuple(xyz[3 * i:3 * i + 3]) for i in range(n)],
            "displacements": [tuple(u[3 * i:3 * i + 3]) for i in range(n)],
            "mesh_lines": lines[:i_data]}   # header, points, cells, cell types


def short(path):
    """File name without .vtk and without the solvers' _deformed ending."""
    return path.stem[:-len("_deformed")] if path.stem.endswith("_deformed") else path.stem


def split_result_name(path):
    """results/DJTLED_fast_NH_001_tension_deformed.vtk -> ('DJTLED_fast', 'NH_001_tension'), by the solver names found
    in src/ and the model names found in models/; (None, name) if neither is recognised."""
    return discover.split_result_name(short(path))


# ------------------------------------------------------------------------------------------------- one comparison
def compare(result, baseline, kind, tolerance, write_csv=True, write_vtk=True):
    """Compare one result with one baseline. Prints the summary, writes the per-point files, returns the numbers."""
    title = f"{short(result)}  vs  {short(baseline)}"
    print(f"\n=== {title} ===")
    res, ref = read_vtk(result), read_vtk(baseline)
    n = len(ref["displacements"])
    out = {"title": title, "result": short(result), "baseline": short(baseline), "kind": kind, "points": n,
           "passed": False, "error": None}
    if len(res["displacements"]) != n:
        out["error"] = f"the result has {len(res['displacements'])} points, the baseline has {n}"
        print(f"  Cannot compare: {out['error']}.")
        return out

    largest, where, squares = [0.0] * 4, [0] * 4, [0.0] * 4   # per column: x, y, z, magnitude
    diffs = []
    for i in range(n):
        d = [res["displacements"][i][k] - ref["displacements"][i][k] for k in range(3)]
        d.append(math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
        for k in range(4):
            if abs(d[k]) > largest[k]:
                largest[k], where[k] = abs(d[k]), i
            squares[k] += d[k] ** 2
        diffs.append(d)
    rmse = [math.sqrt(s / n) for s in squares]
    length = lambda u: math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    ref_max = max(length(u) for u in ref["displacements"])
    res_max = max(length(u) for u in res["displacements"])
    rel_max = 100 * largest[3] / ref_max if ref_max > 0 else (0.0 if largest[3] == 0 else math.inf)
    rel_rmse = 100 * rmse[3] / ref_max if ref_max > 0 else (0.0 if rmse[3] == 0 else math.inf)
    passed = rel_max <= tolerance
    out.update(max=largest, at=where, rmse=rmse, result_max=res_max, baseline_max=ref_max,
               rel_max=rel_max, rel_rmse=rel_rmse, passed=passed)

    labels = ("x", "y", "z", "magnitude")
    print(f"  {n} points; largest displacement: result {res_max:.6g}, baseline {ref_max:.6g}")
    print(f"  {'':24}" + "".join(f"{label:>14}" for label in labels))
    print(f"  {'max difference':24}" + "".join(f"{v:>14.4e}" for v in largest))
    print(f"  {'at point id':24}" + "".join(f"{i:>14}" for i in where))
    print(f"  {'RMSE (whole model)':24}" + "".join(f"{v:>14.4e}" for v in rmse))
    print(f"  relative to the largest baseline displacement: max {rel_max:.4f} %, RMSE {rel_rmse:.4f} %")
    print(f"  {'PASS' if passed else 'FAIL'}: max difference is {rel_max:.4f} % (tolerance {tolerance:g} %)")

    # distribution of the difference magnitude, the worst points and a thinned-out point cloud, for the report
    bins = 20
    top = largest[3]
    counts = [0] * bins
    for d in diffs:
        counts[min(bins - 1, int(d[3] / top * bins)) if top > 0 else 0] += 1
    order = sorted(range(n), key=lambda i: diffs[i][3])
    out["histogram"] = {"edges": [top * b / bins for b in range(bins + 1)], "counts": counts}
    out["worst"] = [[i, *diffs[i]] for i in reversed(order[-10:])]
    keep = order
    if n > MAP_POINTS:   # all of the 2000 worst points, and an even sample of the rest
        rest = order[:-2000]
        keep = rest[::math.ceil(len(rest) / (MAP_POINTS - 2000))] + order[-2000:]
    rounded = lambda v: float(f"{v:.5g}")
    out["map"] = {"shown": len(keep),   # sorted by difference, so the largest are drawn last (on top)
                  "id": keep,
                  "x": [rounded(res["points"][i][0]) for i in keep],
                  "y": [rounded(res["points"][i][1]) for i in keep],
                  "z": [rounded(res["points"][i][2]) for i in keep],
                  "e": [float(f"{diffs[i][3]:.4g}") for i in keep]}

    stem = f"compare_{short(result)}_vs_{short(baseline)}"
    if write_csv or write_vtk:
        RESULTS.mkdir(exist_ok=True)
    if write_csv:
        table = RESULTS / f"{stem}.csv"
        with open(table, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["point_id", "baseline_ux", "baseline_uy", "baseline_uz", "result_ux", "result_uy", "result_uz",
                             "diff_x", "diff_y", "diff_z", "diff_magnitude"])
            for i in range(n):
                writer.writerow([i, *(f"{v:.9g}" for v in (*ref["displacements"][i], *res["displacements"][i], *diffs[i]))])
        out["csv"] = table.name
        print(f"  per-point table: {table}")
    if write_vtk:   # the result's mesh with the difference as point data
        error_map = RESULTS / f"{stem}.vtk"
        with open(error_map, "w") as f:
            mesh = list(res["mesh_lines"])
            mesh[1] = f"difference {short(result)} - {short(baseline)}"[:255]
            f.write("\n".join(mesh) + "\n")
            f.write(f"POINT_DATA {n}\nVECTORS difference float\n")
            f.writelines(f"{d[0]:.7g} {d[1]:.7g} {d[2]:.7g}\n" for d in diffs)
            f.write("SCALARS difference_magnitude float 1\nLOOKUP_TABLE default\n")
            f.writelines(f"{d[3]:.7g}\n" for d in diffs)
            f.write("VECTORS displacement float\n")
            f.writelines(f"{u[0]:.7g} {u[1]:.7g} {u[2]:.7g}\n" for u in res["displacements"])
        out["vtk"] = error_map.name
        print(f"  error map for ParaView: {error_map}")
    return out


# ------------------------------------------------------------------------------------------------- timings
def read_timings():
    """results/timings.csv (written by run.py --benchmark): {model: {"steps", "elements", "runs": {threads: {solver: [ms, ...]}}}}."""
    models = {}
    if not TIMINGS.is_file():
        return models
    with open(TIMINGS, newline="") as f:
        for row in csv.DictReader(f):
            try:
                model = models.setdefault(row["model"], {"steps": int(row["steps"]), "elements": int(row["elements"]), "runs": {}})
                model["runs"].setdefault(int(row["threads"]), {}).setdefault(row["algorithm"], []).append(float(row["computation_ms"]))
            except (KeyError, ValueError):
                continue   # a line that is not a timing
    return models


def speed_of(selected, one):
    """The speed comparison for the selected files: for each model, the timings of the solvers behind the selected
    results, with the baseline's solver first (if the baseline is a reference: the solver with the shortest name).
    Returns (list of models with their timings, list of (solver, model) that have not been timed)."""
    wanted = {}   # model -> solver names
    for path in selected:
        solver, model = split_result_name(path)
        if solver is not None and solver not in wanted.setdefault(model, []):
            wanted[model].append(solver)
    base = split_result_name(one)[0]
    recorded = read_timings()
    timings, missing = [], []
    for model, names in wanted.items():
        runs = recorded.get(model, {}).get("runs", {})
        timed = [name for name in names if any(name in by_solver for by_solver in runs.values())]
        missing += [(name, model) for name in names if name not in timed]
        if not timed:
            continue
        order = discover.baseline_first(timed)
        if base in order:
            order = [base] + [name for name in order if name != base]
        rows = []
        for threads in sorted(runs):
            solvers = {name: {"runs": runs[threads][name], "best": min(runs[threads][name]), "median": statistics.median(runs[threads][name])}
                       for name in order if name in runs[threads]}
            if not solvers:
                continue
            first = solvers.get(order[0])   # speed-up: the baseline solver's fastest run divided by this solver's fastest run
            speedups = {name: first["best"] / t["best"] for name, t in solvers.items() if first and name != order[0] and t["best"] > 0}
            rows.append({"threads": threads, "solvers": solvers, "speedups": speedups})
        timings.append({"model": model, "steps": recorded[model]["steps"], "elements": recorded[model]["elements"],
                        "algorithms": order, "rows": rows})
    return timings, missing


def time_missing(missing):
    """Run the solvers that have not been timed yet (one run each, no export) and add them to results/timings.csv.
    They get the thread counts already recorded for that model, so the rows line up; otherwise the default count."""
    import run   # run.py: builds the solvers and holds the benchmark
    solvers = {s.name: s for s in discover.find_solvers()}
    models = {m.stem: m for m in discover.find_models()}
    recorded = read_timings()
    for model in dict.fromkeys(model for _, model in missing):
        names = [name for name, m in missing if m == model]
        if model not in models or not all(name in solvers for name in names):
            continue   # the model file or a solver source is no longer there: it stays in the "not timed" list
        threads = sorted(recorded.get(model, {}).get("runs", {})) or [discover.default_threads()]
        chosen = [solvers[name] for name in names]
        print(f"\nTiming {', '.join(names)} on {model} (not timed yet) ...", flush=True)
        run.build(chosen)
        run.benchmark(chosen, models[model], threads, repeats=1)


def print_speed(timings, missing):
    print("\n=== Speed (fastest computation time, from results/timings.csv) ===")
    for model in timings:
        names = model["algorithms"]
        column = max(24, max(len(name) for name in names) + 20)
        print(f"  {model['model']}: {model['elements']:,} elements, {model['steps']:,} steps")
        print("    threads" + "".join(f"{name:>{column}}" for name in names))
        for row in model["rows"]:
            line = f"    {row['threads']:>7}"
            for name in names:
                t = row["solvers"].get(name)
                text = "-" if t is None else ms_text(t["best"])
                if name in row["speedups"]:
                    ratio = row["speedups"][name]
                    text += "  (same speed)" if 0.95 <= ratio < 1.05 else f"  ({ratio:.1f}x faster)" if ratio > 1 else f"  ({1 / ratio:.1f}x slower)"
                line += f"{text:>{column}}"
            print(line)
    for model in dict.fromkeys(model for _, model in missing):
        names = [solver for solver, m in missing if m == model]
        print(f"  Not timed on {model}: {', '.join(names)}. Run:  python3 run.py --benchmark -m {model}")
    if not timings and not missing:
        print("  Nothing to time: none of the selected files is a solver result.")


# ------------------------------------------------------------------------------------------------- HTML report
def sci(v):
    return "0" if v == 0 else f"{v:.3e}"


def ms_text(ms):
    return f"{ms / 1000:.1f} s" if ms >= 10000 else f"{ms:.0f} ms"


def status_cell(pair):
    if pair["error"]:
        return '<span class="status"><span class="icon bad" aria-hidden="true">&#10005;</span> Not comparable</span>'
    if pair["passed"]:
        return '<span class="status"><span class="icon good" aria-hidden="true">&#10003;</span> Pass</span>'
    return '<span class="status"><span class="icon bad" aria-hidden="true">&#10005;</span> Fail</span>'


def build_report(pairs, baseline, timings, missing, tolerance):
    e = html.escape
    kind_text = {"reference": "with a reference", "solver": "solver with solver"}
    done = [p for p in pairs if not p["error"]]
    passed = sum(1 for p in done if p["passed"])
    parts = []

    # summary
    rows = []
    for k, p in enumerate(pairs):
        if p["error"]:
            rows.append(f'<tr><td><a href="#pair{k}">{e(p["result"])}</a><br><span class="muted">vs {e(p["baseline"])}</span></td>'
                        f'<td>{kind_text[p["kind"]]}</td><td colspan="4">{e(p["error"])}</td><td>{status_cell(p)}</td></tr>')
            continue
        rows.append(f'<tr><td><a href="#pair{k}">{e(p["result"])}</a><br><span class="muted">vs {e(p["baseline"])}</span></td>'
                    f'<td>{kind_text[p["kind"]]}</td><td class="num">{p["points"]:,}</td><td class="num">{sci(p["max"][3])}</td>'
                    f'<td class="num">{sci(p["rmse"][3])}</td><td class="num">{p["rel_max"]:.4f} %</td><td>{status_cell(p)}</td></tr>')
    parts.append(f'''<section class="card"><h2>Summary</h2>
<p class="lead">Everything is compared with the baseline <strong>{e(baseline)}</strong>.
{passed} of {len(pairs)} comparisons pass. A comparison passes when its largest difference is at most
{tolerance:g} % of the largest displacement in the baseline.</p>
<div class="scroll"><table><thead><tr><th>Comparison</th><th>Kind</th><th class="num">Points</th><th class="num">Max difference</th>
<th class="num">RMSE</th><th class="num">Max, % of largest displacement</th><th>Result</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">Differences are magnitudes of (result &minus; baseline) per point, in the model's length unit.</p></section>''')

    # speed: the solvers behind the selected results
    missing = "".join(f'<p class="lead">Not timed on {e(model)} yet: {e(", ".join(solver for solver, m in missing if m == model))}. '
                      f'Run <code>python3 run.py --benchmark -m {e(model)}</code>, then run <code>compare.py</code> again.</p>'
                      for model in dict.fromkeys(model for _, model in missing))
    if not timings and not missing:
        missing = '<p class="lead">None of the selected files is a solver result, so there is nothing to time.</p>'
    if timings:
        blocks = []
        for m, model in enumerate(timings):
            names, base = model["algorithms"], model["algorithms"][0]
            rows = []
            for row in model["rows"]:
                cells = []
                for name in names:
                    t = row["solvers"].get(name)
                    cells.append(f'<td class="num">{ms_text(t["best"])}</td><td class="num">{ms_text(t["median"])}</td><td class="num">{len(t["runs"])}</td>'
                                 if t else '<td class="num">&ndash;</td><td class="num">&ndash;</td><td class="num">0</td>')
                speedups = "".join(f'<td class="num"><strong>{row["speedups"][name]:.1f}&times;</strong></td>' if name in row["speedups"]
                                   else '<td class="num">&ndash;</td>' for name in names[1:])
                quickest = min((t["best"] for t in row["solvers"].values()), default=None)
                per_step = f'{1000 * quickest / model["steps"]:.1f} &micro;s' if quickest is not None and model["steps"] else "&ndash;"
                rows.append(f'<tr><td class="num">{row["threads"]}</td>{"".join(cells)}{speedups}<td class="num">{per_step}</td></tr>')
            legend = "".join(f'<span><i class="swatch s{i + 1}"></i>{e(name)}{" (baseline)" if i == 0 and len(names) > 1 else ""}</span>'
                             for i, name in enumerate(names[:8]))
            head1 = "".join(f'<th colspan="3">{e(name)}</th>' for name in names)
            head1 += "".join(f'<th class="num" rowspan="2">Speed-up of {e(name)}</th>' for name in names[1:])
            head2 = '<th class="num">Fastest</th><th class="num">Median</th><th class="num">Runs</th>' * len(names)
            blocks.append(f'''<h3>{e(model["model"])} <span class="muted">{model["elements"]:,} elements, {model["steps"]:,} steps</span></h3>
<div class="legend">{legend}</div>
<div class="chart timing" data-model="{m}" role="img" aria-label="Fastest computation time per solver and thread count; the values are in the table below"></div>
<div class="scroll"><table><thead><tr><th class="num" rowspan="2">Threads</th>{head1}<th class="num" rowspan="2">Fastest solver, time per step</th></tr>
<tr>{head2}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>''')
        parts.append(f'''<section class="card"><h2>Speed</h2>
<p class="lead">Computation time of the time loop only (no file reading, no VTK export), from <code>results/timings.csv</code>.
Speed-up is the baseline solver's fastest run divided by the other solver's fastest run at the same thread count.</p>
{"".join(blocks)}{missing}</section>''')
    else:
        parts.append(f'<section class="card"><h2>Speed</h2>{missing}</section>')

    # one section per comparison
    for k, p in enumerate(pairs):
        if p["error"]:
            parts.append(f'<section class="card" id="pair{k}"><h2>{e(p["result"])} <span class="muted">vs {e(p["baseline"])}</span></h2>'
                         f'<p>{status_cell(p)}: {e(p["error"])}.</p></section>')
            continue
        number_rows = "".join(
            f'<tr><th scope="row">{label}</th><td class="num">{sci(p["max"][c])}</td><td class="num">{p["at"][c]}</td><td class="num">{sci(p["rmse"][c])}</td></tr>'
            for c, label in enumerate(("x", "y", "z", "Magnitude")))
        worst_rows = "".join(
            f'<tr><td class="num">{w[0]}</td><td class="num">{sci(w[1])}</td><td class="num">{sci(w[2])}</td><td class="num">{sci(w[3])}</td><td class="num">{sci(w[4])}</td></tr>'
            for w in p["worst"])
        h = p["histogram"]
        hist_rows = "".join(f'<tr><td class="num">{sci(h["edges"][b])} to {sci(h["edges"][b + 1])}</td><td class="num">{c:,}</td></tr>'
                            for b, c in enumerate(h["counts"]))
        files = " &middot; ".join(f'<a href="{e(p[key])}">{label}</a>' for key, label in (("csv", "per-point table (CSV)"), ("vtk", "error map for ParaView (VTK)")) if key in p)
        shown = (f'{p["map"]["shown"]:,} of {p["points"]:,} points drawn (all of the 2,000 with the largest difference, and an even sample of the rest)'
                 if p["map"]["shown"] < p["points"] else f'all {p["points"]:,} points drawn')
        parts.append(f'''<section class="card" id="pair{k}">
<h2>{e(p["result"])} <span class="muted">vs {e(p["baseline"])}</span></h2>
<p class="lead">{status_cell(p)} &nbsp; Largest difference {sci(p["max"][3])}, which is {p["rel_max"]:.4f} % of the largest baseline displacement
({p["baseline_max"]:.6g}); tolerance {tolerance:g} %. RMSE {sci(p["rmse"][3])} ({p["rel_rmse"]:.4f} %).</p>
<div class="grid">
<div><h3>Difference by direction</h3>
<table><thead><tr><th></th><th class="num">Max difference</th><th class="num">At point id</th><th class="num">RMSE</th></tr></thead><tbody>{number_rows}</tbody></table>
<h3>The ten points with the largest difference</h3>
<table><thead><tr><th class="num">Point id</th><th class="num">x</th><th class="num">y</th><th class="num">z</th><th class="num">Magnitude</th></tr></thead><tbody>{worst_rows}</tbody></table>
</div>
<div><h3>How the difference is spread over the points</h3>
<div class="chart histogram" data-pair="{k}" role="img" aria-label="Number of points per range of difference magnitude; the values are in the table below"></div>
<details><summary>Show as a table</summary><table><thead><tr><th class="num">Difference magnitude</th><th class="num">Points</th></tr></thead><tbody>{hist_rows}</tbody></table></details>
</div></div>
<h3>Where the points differ</h3>
<p class="muted">Three views of the deformed model; each dot is a point, darker means a larger difference, and the largest are drawn on top.
{shown}. Hover a dot for its point id.</p>
<div class="maps">
<figure><canvas class="map" data-pair="{k}" data-axes="x,y" width="320" height="320"></canvas><figcaption>x (right), y (up)</figcaption></figure>
<figure><canvas class="map" data-pair="{k}" data-axes="x,z" width="320" height="320"></canvas><figcaption>x (right), z (up)</figcaption></figure>
<figure><canvas class="map" data-pair="{k}" data-axes="z,y" width="320" height="320"></canvas><figcaption>z (right), y (up)</figcaption></figure>
</div>
<div class="scale"><span>0</span><i class="rampbar"></i><span>{sci(p["max"][3])}</span><span class="muted">difference magnitude</span></div>
<p class="muted">{files}</p>
</section>''')

    data = {"pairs": [{"histogram": p.get("histogram"), "map": p.get("map"), "max": (p.get("max") or [0, 0, 0, 0])[3]} for p in pairs],
            "timings": timings}
    page = (TEMPLATE.replace("__GENERATED__", datetime.datetime.now().strftime("%d %B %Y, %H:%M"))
                    .replace("__BODY__", "\n".join(parts))
                    .replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    RESULTS.mkdir(exist_ok=True)
    REPORT.write_text(page, encoding="utf-8")
    return REPORT


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DJ-TLED comparison</title>
<style>
:root { color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --text: #0b0b0b; --text2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --good: #0ca30c; --bad: #d03b3b; --ramp-lo: #cde2fb; --ramp-hi: #0d366b; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --text: #f0efec; --text2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
  --good: #0ca30c; --bad: #d03b3b; --ramp-lo: #0d366b; --ramp-hi: #cde2fb; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--page); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 0 0 8px; } h3 { font-size: 14px; margin: 20px 0 8px; }
h2 .muted, h3 .muted { font-weight: 400; font-size: 13px; }
.muted { color: var(--muted); } .lead { color: var(--text2); margin: 0 0 12px; max-width: 80ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-top: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 8px 32px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
thead th { color: var(--text2); font-weight: 600; border-bottom: 1px solid var(--axis); }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
a { color: inherit; } code { font-size: 12.5px; }
.status { white-space: nowrap; font-weight: 600; } .icon { display: inline-block; width: 1.1em; font-weight: 700; }
.icon.good { color: var(--good); } .icon.bad { color: var(--bad); }
.chart { width: 100%; } .chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart text.value { fill: var(--text2); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; color: var(--text2); font-size: 13px; margin: 4px 0 8px; }
.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.s1 { background: var(--s1); } .s2 { background: var(--s2); } .s3 { background: var(--s3); } .s4 { background: var(--s4); }
.s5 { background: var(--s5); } .s6 { background: var(--s6); } .s7 { background: var(--s7); } .s8 { background: var(--s8); }
.maps { display: flex; flex-wrap: wrap; gap: 16px; } .maps figure { margin: 0; }
.maps canvas { display: block; width: 320px; max-width: 100%; height: auto; border: 1px solid var(--grid); border-radius: 6px; }
figcaption { color: var(--muted); font-size: 12px; margin-top: 4px; }
.scale { display: flex; align-items: center; gap: 8px; margin: 12px 0 4px; font-size: 12px; color: var(--text2); font-variant-numeric: tabular-nums; }
.rampbar { display: inline-block; width: 180px; height: 10px; border-radius: 5px; background: linear-gradient(to right, var(--ramp-lo), var(--ramp-hi)); }
details { margin-top: 8px; } summary { cursor: pointer; color: var(--text2); font-size: 13px; }
#tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; font-size: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.15); max-width: 280px; }
#tip strong { display: block; font-size: 13px; } #tip span { color: var(--text2); display: block; }
.hit { fill: transparent; } .hit:focus { outline: none; }
</style></head>
<body><main>
<h1>DJ-TLED comparison</h1>
<p class="muted">Generated __GENERATED__ by compare.py. All lengths are in the model's length unit; point ids count from 0.</p>
__BODY__
</main>
<div id="tip" hidden></div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const NS = "http://www.w3.org/2000/svg";
const RAMP = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"];
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const sci = v => v === 0 ? "0" : v.toExponential(2);
const msText = ms => ms >= 10000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";

function el(name, attrs, parent) {
  const node = document.createElementNS(NS, name);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  if (parent) parent.appendChild(node);
  return node;
}
function niceStep(top, ticks) {            // a round step giving about `ticks` gridlines up to `top`
  const raw = top / ticks, power = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / power;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * power;
}

// one tooltip for every chart; text is set with textContent only
const tip = document.getElementById("tip");
function showTip(x, y, strong, lines) {
  tip.textContent = "";
  const head = document.createElement("strong"); head.textContent = strong; tip.appendChild(head);
  for (const line of lines) { const s = document.createElement("span"); s.textContent = line; tip.appendChild(s); }
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.max(8, Math.min(innerWidth - w - 8, x + 14)) + "px";
  tip.style.top = Math.max(8, y - h - 12 < 8 ? y + 18 : y - h - 12) + "px";
}
function hideTip() { tip.hidden = true; }
function hover(target, strong, lines) {    // the same details on pointer and on keyboard focus
  target.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, strong, lines));
  target.addEventListener("pointerleave", hideTip);
  target.addEventListener("focus", () => { const r = target.getBoundingClientRect(); showTip(r.left + r.width / 2, r.top, strong, lines); });
  target.addEventListener("blur", hideTip);
}

// column chart: number of points per range of difference magnitude (one series, so no legend)
function drawHistogram(box) {
  const h = DATA.pairs[+box.dataset.pair].histogram;
  if (!h) return;
  const W = 520, H = 230, left = 46, right = 12, top = 12, bottom = 34;
  const plotW = W - left - right, plotH = H - top - bottom, n = h.counts.length;
  const peak = Math.max(1, ...h.counts), step = niceStep(peak, 4), yMax = Math.ceil(peak / step) * step;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  for (let v = 0; v <= yMax + 1e-9; v += step) {
    const y = top + plotH - plotH * v / yMax;
    el("line", { x1: left, x2: W - right, y1: y, y2: y, stroke: cssVar(v === 0 ? "--axis" : "--grid"), "stroke-width": 1 }, svg);
    el("text", { x: left - 6, y: y + 4, "text-anchor": "end" }, svg).textContent = v.toLocaleString();
  }
  const slot = plotW / n, bar = Math.min(24, slot - 2);
  h.counts.forEach((count, b) => {
    const x = left + slot * b + (slot - bar) / 2, height = plotH * count / yMax, y = top + plotH - height;
    if (count > 0) {
      const r = Math.min(4, height);     // rounded top, square at the baseline
      el("path", { d: `M${x},${top + plotH} V${y + r} Q${x},${y} ${x + r},${y} H${x + bar - r} Q${x + bar},${y} ${x + bar},${y + r} V${top + plotH} Z`, fill: cssVar("--s1") }, svg);
    }
    const hit = el("rect", { x: left + slot * b, y: top, width: slot, height: plotH, class: "hit", tabindex: 0 }, svg);
    hover(hit, count.toLocaleString() + (count === 1 ? " point" : " points"), ["difference " + sci(h.edges[b]) + " to " + sci(h.edges[b + 1])]);
  });
  [0, n / 2, n].forEach((b, i) => {
    el("text", { x: left + slot * b, y: H - 16, "text-anchor": ["start", "middle", "end"][i] }, svg).textContent = sci(h.edges[b]);
  });
  el("text", { x: left + plotW / 2, y: H - 2, "text-anchor": "middle" }, svg).textContent = "difference magnitude";
}

// horizontal bars: fastest computation time per thread count, one bar per solver (colours in the legend's order)
function drawTiming(box) {
  const model = DATA.timings[+box.dataset.model];
  const series = model.algorithms.slice(0, 8).map((name, i) => [name, "--s" + (i + 1)]);
  const W = 720, left = 86, right = 70, top = 6, bar = 16, gap = 2, rowH = series.length * (bar + gap) + 10;
  const H = top + rowH * model.rows.length + 24, plotW = W - left - right;
  const peak = Math.max(1, ...model.rows.flatMap(row => series.map(s => row.solvers[s[0]] ? row.solvers[s[0]].best : 0)));
  const step = niceStep(peak, 4), xMax = Math.ceil(peak / step) * step;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  for (let v = 0; v <= xMax + 1e-9; v += step) {
    const x = left + plotW * v / xMax;
    el("line", { x1: x, x2: x, y1: top, y2: H - 22, stroke: cssVar(v === 0 ? "--axis" : "--grid"), "stroke-width": 1 }, svg);
    el("text", { x: x, y: H - 8, "text-anchor": "middle" }, svg).textContent = msText(v);
  }
  model.rows.forEach((row, r) => {
    const y0 = top + rowH * r + 5;
    el("text", { x: left - 8, y: y0 + (rowH - 10) / 2 + 3, "text-anchor": "end", class: "value" }, svg).textContent = row.threads + (row.threads === 1 ? " thread" : " threads");
    series.forEach((s, i) => {
      const t = row.solvers[s[0]];
      if (!t) return;
      const y = y0 + i * (bar + gap), width = Math.max(1, plotW * t.best / xMax), rr = Math.min(4, width);
      el("path", { d: `M${left},${y} H${left + width - rr} Q${left + width},${y} ${left + width},${y + rr} V${y + bar - rr} Q${left + width},${y + bar} ${left + width - rr},${y + bar} H${left} Z`, fill: cssVar(s[1]) }, svg);
      el("text", { x: left + width + 6, y: y + bar - 4, class: "value" }, svg).textContent = msText(t.best);
      const hit = el("rect", { x: left, y: y - gap / 2, width: plotW + right, height: bar + gap, class: "hit", tabindex: 0 }, svg);
      hover(hit, msText(t.best) + " fastest", [s[0] + ", " + row.threads + (row.threads === 1 ? " thread" : " threads"), "median " + msText(t.median) + " over " + t.runs.length + (t.runs.length === 1 ? " run" : " runs")]);
    });
  });
}

// point cloud coloured by difference magnitude (one hue; light = small, dark = large; reversed on a dark page)
function drawMap(canvas) {
  const pair = DATA.pairs[+canvas.dataset.pair], map = pair.map;
  if (!map) return;
  const [ax, ay] = canvas.dataset.axes.split(","), xs = map[ax], ys = map[ay];
  const ratio = devicePixelRatio || 1, size = 320, pad = 14;
  canvas.width = canvas.height = size * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, size, size);
  const min = a => a.reduce((m, v) => Math.min(m, v), Infinity), max = a => a.reduce((m, v) => Math.max(m, v), -Infinity);
  const x0 = min(xs), x1 = max(xs), y0 = min(ys), y1 = max(ys);
  const scale = (size - 2 * pad) / Math.max(x1 - x0, y1 - y0, 1e-30);
  const ox = (size - (x1 - x0) * scale) / 2, oy = (size - (y1 - y0) * scale) / 2;
  const ramp = darkQuery.matches ? RAMP.slice().reverse() : RAMP;
  const radius = xs.length > 5000 ? 1.6 : 2.6, px = new Float32Array(xs.length), py = new Float32Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    px[i] = ox + (xs[i] - x0) * scale; py[i] = size - oy - (ys[i] - y0) * scale;
    const t = pair.max > 0 ? map.e[i] / pair.max : 0;
    ctx.fillStyle = ramp[Math.min(ramp.length - 1, Math.floor(t * ramp.length))];
    ctx.beginPath(); ctx.arc(px[i], py[i], radius, 0, 6.2832); ctx.fill();
  }
  canvas.onpointermove = ev => {             // nearest point within reach; the largest difference wins a tie
    const box = canvas.getBoundingClientRect(), k = size / box.width, mx = (ev.clientX - box.left) * k, my = (ev.clientY - box.top) * k;
    let best = -1, bestD = 144;
    for (let i = xs.length - 1; i >= 0; i--) {
      const d = (px[i] - mx) * (px[i] - mx) + (py[i] - my) * (py[i] - my);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best < 0) return hideTip();
    showTip(ev.clientX, ev.clientY, "difference " + sci(map.e[best]), ["point id " + map.id[best], ax + " " + xs[best] + ", " + ay + " " + ys[best]]);
  };
  canvas.onpointerleave = hideTip;
}

function drawAll() {
  document.querySelectorAll(".chart").forEach(box => { box.textContent = ""; });
  document.querySelectorAll(".histogram").forEach(drawHistogram);
  document.querySelectorAll(".timing").forEach(drawTiming);
  document.querySelectorAll("canvas.map").forEach(drawMap);
}
drawAll();
darkQuery.addEventListener("change", drawAll);
</script>
</body></html>
'''


# ------------------------------------------------------------------------------------------------- choosing what to compare
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


def find_file(name, candidates, what):
    """Accept a name with or without '_deformed' / '.vtk', or a path to a .vtk file."""
    for c in candidates:
        if name in (c.name, c.stem, short(c)):
            return c
    path = Path(name).expanduser()
    if path.is_file():
        return path.resolve()
    sys.exit(f"Unknown {what} '{name}'. Available: {', '.join(short(c) for c in candidates) or 'none'}")


def ask_several(title, options):
    """Print a numbered menu and return the indexes of the chosen options (one or more numbers, e.g. 1,3)."""
    print(f"\n{title}")
    for i, text in enumerate(options, 1):
        print(f"  {i}) {text}")
    while True:
        answer = input(f"Enter the numbers, separated by commas (1-{len(options)}): ").replace(",", " ").split()
        if answer and all(a.isdigit() and 1 <= int(a) <= len(options) for a in answer):
            return list(dict.fromkeys(int(a) - 1 for a in answer))
        print("  Please type one or more of the numbers above, for example 1,2.")


def main():
    parser = argparse.ArgumentParser(description="Compare results with one baseline, in errors and in speed, and build an HTML report.")
    parser.add_argument("-o", "--one", help="the baseline everything is compared with: a result or reference name, or a path to a .vtk file")
    parser.add_argument("-w", "--with", dest="others", nargs="+", metavar="NAME",
                        help="what to compare with the baseline: one name, several names, or 'all' "
                             "(everything with the same number of points as the baseline)")
    parser.add_argument("--tol", type=float, default=0.5, help="pass if the largest difference is at most this many percent "
                                                               "of the largest baseline displacement (default 0.5)")
    parser.add_argument("--no-timing", action="store_true", help="do not run solvers that have not been timed yet (their speed is then missing)")
    parser.add_argument("--no-csv", action="store_true", help="do not save the per-point tables")
    parser.add_argument("--no-vtk", action="store_true", help="do not save the error maps for ParaView")
    parser.add_argument("--no-open", action="store_true", help="do not open the HTML report in the browser")
    args = parser.parse_args()
    if not args.tol >= 0:
        parser.error("--tol must be 0 or more")

    results, references = discover.find_results(), discover.find_references()
    everything = references + results
    if not results:
        sys.exit(f"No results found in {RESULTS}. Run a solver first:  python3 run.py")
    label = lambda path: f"{short(path)}  ({'reference' if path in references else 'result'}, {discover.point_count(path):,} points)"

    # the one baseline
    if args.one:
        one = find_file(args.one, everything, "result or reference")
    else:
        one = everything[ask("Which one is the baseline that the others are compared with?", [label(path) for path in everything])]
    candidates = [path for path in everything if path != one and discover.point_count(path) == discover.point_count(one)]

    # what to compare with it: one other, several others, or all others
    if args.others:
        if [name.lower() for name in args.others] == ["all"]:
            others = candidates
        else:
            others = list(dict.fromkeys(find_file(name, everything, "result or reference") for name in args.others))
    else:
        if not candidates:
            sys.exit(f"Nothing else has the same number of points as {short(one)} ({discover.point_count(one):,}), so there is nothing to compare it with.")
        mode = ask(f"Compare {short(one)} with:", ["one other", "several others", f"all others ({len(candidates)})"])
        if mode == 0:
            others = [candidates[ask("Which one?", [label(path) for path in candidates])]]
        elif mode == 1:
            others = [candidates[i] for i in ask_several("Which ones?", [label(path) for path in candidates])]
        else:
            others = candidates
    others = [path for path in others if path != one]
    if not others:
        sys.exit(f"Nothing to compare {short(one)} with.")

    # errors: every selected item against the baseline
    is_solver_result = lambda path: split_result_name(path)[0] is not None and path not in references
    done = [compare(other, one, "solver" if is_solver_result(other) and is_solver_result(one) else "reference", args.tol,
                    write_csv=not args.no_csv, write_vtk=not args.no_vtk) for other in others]
    # speed: the solvers behind the selected results
    timings, missing = speed_of([one] + others, one)
    if missing and not args.no_timing:
        listed = ", ".join(f"{solver} on {model}" for solver, model in missing)
        if (args.one and args.others) or input(f"\nNot timed yet: {listed}. Run them now to get their speed? [Y/n]: ").strip().lower() in ("", "y", "yes"):
            time_missing(missing)
            timings, missing = speed_of([one] + others, one)
    print_speed(timings, missing)

    report = build_report(done, short(one), timings, missing, args.tol)
    failed = [p for p in done if not p["passed"]]
    print(f"\n{len(done) - len(failed)} of {len(done)} comparisons with {short(one)} pass (tolerance {args.tol:g} %).")
    print(f"HTML report: {report}")
    if not args.no_open:
        try:
            webbrowser.open(report.as_uri())
        except Exception:   # no browser available: the path printed above is enough
            pass
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nCancelled.")
