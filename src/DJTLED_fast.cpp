/*
MIT License

Copyright (c) 2022 Jinao Zhang
Copyright (c) 2026 Baiqi Cheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// DJTLED_fast.cpp: a faster version of DJTLED.cpp (the DJ-TLED code by Jinao Zhang), written by Baiqi Cheng.
// Same algorithm and input format; build it together with vtk_export.cpp (run.py does that).
// Compile-time settings (override with -DNAME=value):
#ifndef SOLVER_NAME       // the name in front of the output files; run.py sets it to this file's name when it compiles
#define SOLVER_NAME "DJTLED_fast"
#endif
#ifndef FAST_W            // elements per block = SIMD lanes of the element kernel (1: scalar kernel)
#define FAST_W 8
#endif
#ifndef FAST_SPIN         // 1: waiting threads spin instead of sleeping between the two barriers of a step
#define FAST_SPIN 1
#endif
#ifndef FAST_PCORES       // 1: Apple silicon, one thread per performance core unless --threads or OMP_NUM_THREADS is set
#define FAST_PCORES 1
#endif
#ifndef J23_TOL           // J^(-2/3): use the warm start only if |J^2 y^3 - 1| < J23_TOL ...
#define J23_TOL 1e-2f
#endif
#ifndef J23_ITERS         // ... refined by this many Newton iterations (2 iterations at 1e-2: relative error < 1e-8)
#define J23_ITERS 2
#endif
#ifndef FAST_POWF         // 1: ablation — use powf for J^(-2/3) in every element instead of the warm-started iteration
#define FAST_POWF 0
#endif
#ifndef FAST_PERSTEP      // 1: ablation — open a parallel region per step instead of one team for the whole time loop
#define FAST_PERSTEP 0
#endif
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include "vtk_export.h"
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
using namespace std;

constexpr int W = FAST_W;

/*------------------------------------------------------------------------------------------------- input model */
struct DispBC { int dir; float mag; vector<unsigned int> nodes; };
struct Model
{
    string                       fname, ele_type, material_type;
    unsigned int                 node_begin_index = 0, ele_begin_index = 0;
    vector<float>                X;                   // nodal coordinates, 3 per node (input numbering)
    vector<array<unsigned, 4>>   tets;                // element nodes (input numbering, from 0)
    float                        mu = 0.f, K = 0.f, rho = 0.f;   // NH material (Mu, K) and density, same for all elements
    vector<DispBC>               disp;                // <Disp>, in file order (a later entry overrides)
    vector<pair<int, vector<unsigned int>>> fix;      // <FixP>, dir 0..2
    vector<pair<int, float>>     grav;                // <Gravity>
    size_t                       num_BCs = 0, num_steps = 0;
    float                        alpha = 0.f, dt = 0.f, total_t = 0.f;
    size_t nodes() const { return X.size() / 3; }
};

static bool readError(FILE* file, const string& msg)
{
    cerr << "\n\tError: " << msg << endl;
    if (file != nullptr) { fclose(file); }
    return false;
}

static bool readModel(const char* fname, Model& m)
{
    FILE* file = fopen(fname, "r");
    if (file == nullptr) { cerr << "\n\tError: cannot open file: " << fname << endl; return false; }
    m.fname = fname;
    char buffer[256];
    unsigned int idx(0); float x(0.f), y(0.f), z(0.f);
    while (fscanf(file, "%u %f %f %f", &idx, &x, &y, &z) == 4) // internally, node index starts at 0
    {
        if (m.X.empty()) { m.node_begin_index = idx; }
        if (idx - m.node_begin_index != m.nodes()) { return readError(file, "node indices must be consecutive."); }
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) { return readError(file, "node coordinates must be finite."); }
        m.X.push_back(x); m.X.push_back(y); m.X.push_back(z);
    }
    if (m.X.empty()) { return readError(file, "no nodes found."); }
    if (fscanf(file, "%255s", buffer) != 1) { return readError(file, "missing material type."); }
    m.material_type = buffer;
    if (m.material_type != "NH") { return readError(file, "unsupported material type: " + m.material_type); }
    if (fscanf(file, "%f %f", &m.mu, &m.K) != 2) { return readError(file, "NH material needs two values (Mu K)."); }
    if (!std::isfinite(m.mu) || !std::isfinite(m.K)) { return readError(file, "NH material values must be finite."); }
    if (!(m.mu > 0.f) || !(m.K > 0.f)) { return readError(file, "NH material needs positive Mu and K."); }
    if (fscanf(file, "%255s %f", buffer, &m.rho) != 2 || string(buffer) != "Density") { return readError(file, "missing or misspelled Density line."); }
    if (!(m.rho > 0.f) || !std::isfinite(m.rho)) { return readError(file, "Density must be a positive finite number."); }
    if (fscanf(file, "%255s", buffer) != 1) { return readError(file, "missing element type."); }
    m.ele_type = buffer;
    if (m.ele_type != "T4") { return readError(file, "unsupported element type: " + m.ele_type + " (only T4 is supported)."); }
    const size_t num_nodes = m.nodes();
    unsigned int n_idx[4] = { 0, 0, 0, 0 };
    while (fscanf(file, "%u %u %u %u %u", &idx, &n_idx[0], &n_idx[1], &n_idx[2], &n_idx[3]) == 5) // internally, ele index starts at 0
    {
        if (m.tets.empty()) { m.ele_begin_index = idx; }
        if (idx - m.ele_begin_index != m.tets.size()) { return readError(file, "element indices must be consecutive."); }
        for (unsigned int& n : n_idx)
        {
            if (n < m.node_begin_index || n - m.node_begin_index >= num_nodes) { return readError(file, "element refers to unknown node " + to_string(n) + "."); }
            n -= m.node_begin_index;
        }
        m.tets.push_back({ n_idx[0], n_idx[1], n_idx[2], n_idx[3] });
    }
    if (m.tets.empty()) { return readError(file, "no elements found."); }
    auto readNodes = [&](vector<unsigned int>& list)
    {
        unsigned int n(0);
        while (fscanf(file, "%u", &n) == 1)
        {
            if (n < m.node_begin_index || n - m.node_begin_index >= num_nodes) { return false; }
            list.push_back(n - m.node_begin_index);
        }
        return true;
    };
    auto dirOf = [](const string& s) { return s == "x" ? 0 : s == "y" ? 1 : s == "z" ? 2 : s == "all" ? 3 : -1; };
    bool found_end(false);
    while (fscanf(file, "%255s", buffer) == 1)
    {
        const string BC_type(buffer);
        if (BC_type == "<Disp>") // Displacements
        {
            float u(0.f);
            if (fscanf(file, "%255s %f", buffer, &u) != 2) { return readError(file, "<Disp> needs a direction and a magnitude."); }
            const int dir = dirOf(buffer);
            if (dir < 0 || dir > 2) { return readError(file, "unknown <Disp> direction '" + string(buffer) + "' (use x, y or z)."); }
            if (!std::isfinite(u)) { return readError(file, "<Disp> magnitude must be finite."); }
            DispBC bc{ dir, u, {} };
            if (!readNodes(bc.nodes)) { return readError(file, "<Disp> refers to an unknown node."); }
            if (bc.nodes.empty()) { return readError(file, "<Disp> must name at least one node."); }
            m.disp.push_back(bc);
            m.num_BCs++;
        }
        else if (BC_type == "<FixP>") // Fixed positions
        {
            if (fscanf(file, "%255s", buffer) != 1) { return readError(file, "<FixP> needs a direction."); }
            const int d = dirOf(buffer);
            if (d < 0) { return readError(file, "unknown <FixP> direction '" + string(buffer) + "' (use x, y, z or all)."); }
            vector<unsigned int> list;
            if (!readNodes(list)) { return readError(file, "<FixP> refers to an unknown node."); }
            if (list.empty()) { return readError(file, "<FixP> must name at least one node."); }
            if (d >= 0 && d <= 2) { m.fix.push_back({ d, list }); }
            if (d == 3) { for (int k = 0; k < 3; k++) { m.fix.push_back({ k, list }); } }
            m.num_BCs++;
        }
        else if (BC_type == "<Gravity>") // Gravity
        {
            float g(0.f);
            if (fscanf(file, "%255s %f", buffer, &g) != 2) { return readError(file, "<Gravity> needs a direction and a magnitude."); }
            const int d = dirOf(buffer);
            if (d < 0 || d > 2) { return readError(file, "unknown <Gravity> direction '" + string(buffer) + "' (use x, y or z)."); }
            if (!std::isfinite(g)) { return readError(file, "<Gravity> magnitude must be finite."); }
            m.grav.push_back({ d, g });
            m.num_BCs++;
        }
        else if (BC_type == "</BC>") { found_end = true; break; }
        else { return readError(file, "unknown boundary-condition tag '" + BC_type + "' (expected <Disp>, <FixP>, <Gravity> or </BC>)."); }
    }
    if (!found_end) { return readError(file, "missing </BC>."); }
    if (fscanf(file, "%255s %f", buffer, &m.alpha)   != 2 || string(buffer) != "Damping")   { return readError(file, "missing or misspelled Damping line."); }
    if (fscanf(file, "%255s %f", buffer, &m.dt)      != 2 || string(buffer) != "TimeStep")  { return readError(file, "missing or misspelled TimeStep line."); }
    if (fscanf(file, "%255s %f", buffer, &m.total_t) != 2 || string(buffer) != "TotalTime") { return readError(file, "missing or misspelled TotalTime line."); }
    fclose(file);
    if (!(m.dt > 0.f) || !(m.total_t > 0.f) || !std::isfinite(m.dt) || !std::isfinite(m.total_t) || !std::isfinite(m.alpha) || !std::isfinite(m.total_t / m.dt))
    { return readError(nullptr, "Damping, TimeStep and TotalTime must be finite, and TimeStep and TotalTime positive."); }
    {
        // single-precision division can round a clean ratio (e.g. 0.05 / 0.000025) just above its integer, so
        // ceil would add one extra step; snap ratios that are within float noise of an integer, then take ceil.
        const double ratio = (double)m.total_t / (double)m.dt;
        const double nearest = std::llround(ratio);
        const double adjusted = (std::fabs(ratio - nearest) <= 1e-6 * std::max(1.0, std::fabs(ratio))) ? nearest : ratio;
        m.num_steps = (size_t)std::ceil(adjusted);
        if (m.num_steps == 0) { m.num_steps = 1; }
    }
    return true;
}

/*------------------------------------------------------------------------------------------------- solver data */
// W elements per block, each array indexed [component][lane] so the kernel loads whole SIMD vectors
struct alignas(W >= 4 ? 64 : 4) EBlock
{
    uint32_t n[4][W];        // first DOF (3 * node) of the 4 nodes
    float    J0[9][W];       // initial Jacobian, row-major
    float    P[6][W];        // mu Vol invJ0^T invJ0 (symmetric: 00 11 22 01 02 12) = DJTLED's m_I1m, also gives I1
    float    invDetJ0[W];
    float    KV[W];          // K Vol
};
// Element nodal forces are stored as one flat array: block b occupies [b * FSTRIDE, (b+1) * FSTRIDE),
// component c of lane l at [b * FSTRIDE + c * W + l]. This keeps all indexing inside a single vector<float>.
constexpr size_t FSTRIDE = 12 * (size_t)W;   // floats per block (12 force components x W lanes)

enum : uint8_t { DOF_FREE = 0, DOF_FIXED = 1, DOF_DISP = 2 };
enum { STOP_NAN = 1, STOP_INVERTED = 2 };   // why the time loop stopped: a displacement became NaN / an element inverted (J <= 0)

struct Solver
{
    size_t                 nn = 0, ne = 0, nb = 0;
    vector<uint32_t>       newOfOld, oldOfNew;            // node numbering
    vector<EBlock>         blocks;
    vector<float>          Fe;                            // flat force buffer, nb * FSTRIDE floats
    vector<float>          J23;                           // J^(-2/3) warm starts, nb * W
    vector<uint32_t>       nptr, noff;                    // node i: force offsets noff[nptr[i] .. nptr[i+1])
    vector<float>          c1, c2, c3, fext, dmag;        // per DOF (solver numbering)
    vector<uint8_t>        kind;
    vector<float>          U[3];
    float*                 U_curr = nullptr;              // final displacements, after the run
    long long              fallbacks = 0;
};

// reverse Cuthill-McKee ordering of the node graph; returns oldOfNew
static vector<uint32_t> rcmOrder(size_t nn, const vector<array<unsigned, 4>>& tets)
{
    vector<uint32_t> deg(nn + 1, 0);
    for (const auto& t : tets) { for (int a = 0; a < 4; a++) { deg[t[a] + 1] += 3; } }
    for (size_t i = 0; i < nn; i++) { deg[i + 1] += deg[i]; }
    vector<uint32_t> adj(deg[nn]), fill(deg.begin(), deg.end() - 1);
    for (const auto& t : tets) { for (int a = 0; a < 4; a++) { for (int b = 0; b < 4; b++) { if (a != b) { adj[fill[t[a]]++] = t[b]; } } } }
    vector<uint32_t> ptr(nn + 1, 0), nbr;   // unique neighbours
    nbr.reserve(adj.size() / 2);
    for (size_t i = 0; i < nn; i++)
    {
        auto b = adj.begin() + deg[i], e = adj.begin() + deg[i + 1];
        sort(b, e); e = unique(b, e);
        nbr.insert(nbr.end(), b, e); ptr[i + 1] = (uint32_t)nbr.size();
    }
    auto degree = [&](uint32_t v) { return ptr[v + 1] - ptr[v]; };
    vector<uint32_t> order; order.reserve(nn);
    vector<int> level(nn, -1);
    vector<char> done(nn, 0);
    auto bfsLast = [&](uint32_t s, vector<uint32_t>& seen) // BFS from s; returns a min-degree node of the last level
    {
        seen.clear(); seen.push_back(s); level[s] = 0;
        for (size_t h = 0; h < seen.size(); h++)
        {
            const uint32_t v = seen[h];
            for (uint32_t k = ptr[v]; k < ptr[v + 1]; k++) { const uint32_t w = nbr[k]; if (level[w] < 0) { level[w] = level[v] + 1; seen.push_back(w); } }
        }
        const int last = level[seen.back()];
        uint32_t best = seen.back();
        for (uint32_t v : seen) { if (level[v] == last && degree(v) < degree(best)) { best = v; } }
        for (uint32_t v : seen) { level[v] = -1; }
        return best;
    };
    vector<uint32_t> seen, cand;
    for (uint32_t s0 = 0; s0 < nn; s0++)
    {
        if (done[s0]) { continue; }
        uint32_t s = bfsLast(s0, seen);          // pseudo-peripheral start node
        s = bfsLast(s, seen);
        const size_t first = order.size();
        order.push_back(s); done[s] = 1;
        for (size_t h = first; h < order.size(); h++)
        {
            const uint32_t v = order[h];
            cand.clear();
            for (uint32_t k = ptr[v]; k < ptr[v + 1]; k++) { if (!done[nbr[k]]) { cand.push_back(nbr[k]); done[nbr[k]] = 1; } }
            sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) { return degree(a) != degree(b) ? degree(a) < degree(b) : a < b; });
            order.insert(order.end(), cand.begin(), cand.end());
        }
        reverse(order.begin() + first, order.end());
    }
    return order;
}

// Returns false if an element of the input mesh is inverted or flat (det J0 <= 0).
static bool setup(const Model& m, Solver& s, const bool reorder)
{
    s.nn = m.nodes(); s.ne = m.tets.size(); s.nb = (s.ne + W - 1) / W;
    if (s.ne > (size_t)UINT32_MAX / 12) { cerr << "\n\tError: mesh is too large for the internal 32-bit counters (adjacency counts 12 entries per element)." << endl; return false; }
    // node and element order
    if (reorder) { s.oldOfNew = rcmOrder(s.nn, m.tets); }
    else { s.oldOfNew.resize(s.nn); for (size_t i = 0; i < s.nn; i++) { s.oldOfNew[i] = (uint32_t)i; } }
    s.newOfOld.assign(s.nn, 0);
    for (size_t i = 0; i < s.nn; i++) { s.newOfOld[s.oldOfNew[i]] = (uint32_t)i; }
    vector<uint32_t> eorder(s.ne);
    for (size_t e = 0; e < s.ne; e++) { eorder[e] = (uint32_t)e; }
    if (reorder)
    {
        vector<uint64_t> key(s.ne);
        for (size_t e = 0; e < s.ne; e++)
        {
            uint32_t lo = UINT32_MAX, hi = 0;
            for (int a = 0; a < 4; a++) { lo = min(lo, s.newOfOld[m.tets[e][a]]); hi = max(hi, s.newOfOld[m.tets[e][a]]); }
            key[e] = ((uint64_t)lo << 32) | hi;
        }
        stable_sort(eorder.begin(), eorder.end(), [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
    }
    // element blocks (precomputed in double, stored in float); padding lanes are inert (J0 = I, zero stiffness)
    s.blocks.assign(s.nb, EBlock{});
    s.Fe.assign((size_t)s.nb * FSTRIDE, 0.f);
    s.J23.assign(s.nb * W, 1.f);
    vector<double> nodal_mass(s.nn, 0.0), gforce(3 * s.nn, 0.0);
    vector<uint32_t> cnt(s.nn + 1, 0);
    for (size_t b = 0; b < s.nb; b++)
    {
        for (int l = 0; l < W; l++)
        {
            EBlock& B = s.blocks[b];
            for (int a = 0; a < 4; a++) { B.n[a][l] = 0; }
            for (int k = 0; k < 9; k++) { B.J0[k][l] = (k % 4 == 0) ? 1.f : 0.f; }
            for (int k = 0; k < 6; k++) { B.P[k][l] = 0.f; }
            B.invDetJ0[l] = 1.f; B.KV[l] = 0.f;
        }
    }
    for (size_t k = 0; k < s.ne; k++)
    {
        const size_t e = eorder[k], b = k / W; const int l = (int)(k % W);
        EBlock& B = s.blocks[b];
        double x[4][3], J0[3][3], inv[3][3];
        for (int a = 0; a < 4; a++)
        {
            const uint32_t v = s.newOfOld[m.tets[e][a]];
            B.n[a][l] = 3 * v; cnt[v + 1]++;
            for (int d = 0; d < 3; d++) { x[a][d] = m.X[3 * (size_t)m.tets[e][a] + d]; }
        }
        for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { J0[i][j] = x[i + 1][j] - x[0][j]; } }
        inv[0][0] = J0[1][1] * J0[2][2] - J0[1][2] * J0[2][1]; inv[0][1] = J0[0][2] * J0[2][1] - J0[0][1] * J0[2][2]; inv[0][2] = J0[0][1] * J0[1][2] - J0[0][2] * J0[1][1];
        inv[1][0] = J0[1][2] * J0[2][0] - J0[1][0] * J0[2][2]; inv[1][1] = J0[0][0] * J0[2][2] - J0[0][2] * J0[2][0]; inv[1][2] = J0[0][2] * J0[1][0] - J0[0][0] * J0[1][2];
        inv[2][0] = J0[1][0] * J0[2][1] - J0[1][1] * J0[2][0]; inv[2][1] = J0[0][1] * J0[2][0] - J0[0][0] * J0[2][1]; inv[2][2] = J0[0][0] * J0[1][1] - J0[0][1] * J0[1][0];
        const double det = J0[0][0] * inv[0][0] + J0[1][0] * inv[0][1] + J0[2][0] * inv[0][2];
        if (!(det > 0.0))
        {
            cerr << "\n\tError: element " << e + m.ele_begin_index << " is inverted or flat (Jacobian determinant " << det
                 << " is not positive): check the order of its four nodes." << endl;
            return false;
        }
        for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { inv[i][j] /= det; } }
        const double vol = det / 6.0, muV = (double)m.mu * vol;
        double Bm[3][3];   // invJ0^T invJ0
        for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { Bm[i][j] = inv[0][i] * inv[0][j] + inv[1][i] * inv[1][j] + inv[2][i] * inv[2][j]; } }
        for (int i = 0; i < 9; i++) { B.J0[i][l] = (float)J0[i / 3][i % 3]; }
        B.P[0][l] = (float)(muV * Bm[0][0]); B.P[1][l] = (float)(muV * Bm[1][1]); B.P[2][l] = (float)(muV * Bm[2][2]);
        B.P[3][l] = (float)(muV * Bm[0][1]); B.P[4][l] = (float)(muV * Bm[0][2]); B.P[5][l] = (float)(muV * Bm[1][2]);
        B.invDetJ0[l] = (float)(1.0 / det);
        B.KV[l] = (float)((double)m.K * vol);
        const double mass = (double)m.rho * vol;
        for (int a = 0; a < 4; a++)
        {
            const uint32_t v = s.newOfOld[m.tets[e][a]];
            nodal_mass[v] += mass / 4.0;
            for (const auto& g : m.grav) { gforce[3 * (size_t)v + g.first] += mass * g.second / 4.0; }
        }
    }
    // reject isolated nodes (zero mass would make the central-difference constants divide by zero)
    for (size_t i = 0; i < s.nn; i++) { if (!(nodal_mass[i] > 0.0)) { cerr << "\n\tError: node " << s.oldOfNew[i] + m.node_begin_index << " is not part of any element (zero mass)." << endl; return false; } }
    // node -> element force offsets (CSR, elements in solver order)
    s.nptr.assign(s.nn + 1, 0);
    for (size_t i = 0; i < s.nn; i++) { s.nptr[i + 1] = s.nptr[i] + cnt[i + 1]; }
    s.noff.assign(s.nptr[s.nn], 0);
    vector<uint32_t> pos(s.nptr.begin(), s.nptr.end() - 1);
    for (size_t k = 0; k < s.ne; k++)
    {
        const size_t b = k / W; const int l = (int)(k % W);
        for (int a = 0; a < 4; a++) { const uint32_t v = s.blocks[b].n[a][l] / 3; s.noff[pos[v]++] = (uint32_t)(b * FSTRIDE + 3 * a * W + l); }
    }
    // per-DOF constants (same float formulas as DJTLED.cpp) and BC codes
    const size_t nd = 3 * s.nn;
    s.c1.assign(nd, 0.f); s.c2.assign(nd, 0.f); s.c3.assign(nd, 0.f); s.fext.assign(nd, 0.f); s.dmag.assign(nd, 0.f); s.kind.assign(nd, DOF_FREE);
    for (size_t i = 0; i < nd; i++)
    {
        const float mass = (float)nodal_mass[i / 3];
        s.c1[i] = 1.f / (m.alpha * mass / 2.f / m.dt + mass / m.dt / m.dt);
        s.c2[i] = 2.f * mass * s.c1[i] / m.dt / m.dt;
        s.c3[i] = m.alpha * mass * s.c1[i] / 2.f / m.dt - s.c2[i] / 2.f;
        s.fext[i] = (float)gforce[i];
    }
    for (const auto& f : m.fix) { for (unsigned int n : f.second) { s.kind[3 * (size_t)s.newOfOld[n] + f.first] = DOF_FIXED; } }
    for (const auto& d : m.disp) { for (unsigned int n : d.nodes) { const size_t di = 3 * (size_t)s.newOfOld[n] + d.dir; s.dmag[di] = d.mag; s.kind[di] = DOF_DISP; } }   // Disp wins over FixP; a zero magnitude still constrains the DOF
    for (auto& u : s.U) { u.assign(nd, 0.f); }
    for (size_t i = 0; i < nd; i++) { s.U[0][i] = 0.5f * m.dt * m.dt * s.fext[i] / (float)nodal_mass[i / 3]; }   // start from rest under the initial acceleration
    return true;
}

/*------------------------------------------------------------------------------------------------- kernels */
#define FORCE_INLINE inline __attribute__((always_inline))
// Element nodal internal forces of lane l of a block (DJTLED.cpp Table 3, NH), written to f[3 * local node + xyz][l].
// EXACT = 0: J^(-2/3) from the warm start; returns 1 if the warm start was too far off (or NaN) or the element is
//            inverted (J <= 0; the warm start works with J^2 and would not notice the sign), so that EXACT = 1 follows.
// EXACT = 1: J^(-2/3) = powf(J, -2/3); returns 2 if the element is inverted (J <= 0 or NaN), else 0.
template <int EXACT>
static FORCE_INLINE int laneForces(const EBlock& __restrict E, const int l, const float (&u)[12][W], float* __restrict y_cache, float* __restrict f)
{
    // Eq. (11): Jt = J0 + displacement differences
    const float a00 = E.J0[0][l] - u[0][l] + u[3][l], a01 = E.J0[1][l] - u[1][l] + u[4][l],  a02 = E.J0[2][l] - u[2][l] + u[5][l];
    const float a10 = E.J0[3][l] - u[0][l] + u[6][l], a11 = E.J0[4][l] - u[1][l] + u[7][l],  a12 = E.J0[5][l] - u[2][l] + u[8][l];
    const float a20 = E.J0[6][l] - u[0][l] + u[9][l], a21 = E.J0[7][l] - u[1][l] + u[10][l], a22 = E.J0[8][l] - u[2][l] + u[11][l];
    // Eq. (17)
    const float g11 = a00 * a00 + a01 * a01 + a02 * a02, g22 = a10 * a10 + a11 * a11 + a12 * a12, g33 = a20 * a20 + a21 * a21 + a22 * a22;
    const float g12 = a00 * a10 + a01 * a11 + a02 * a12, g13 = a00 * a20 + a01 * a21 + a02 * a22, g23 = a10 * a20 + a11 * a21 + a12 * a22;
    // adjugate of Jt (= det * invJt) and det
    const float i00 = a11 * a22 - a12 * a21, i01 = a02 * a21 - a01 * a22, i02 = a01 * a12 - a02 * a11;
    const float i10 = a12 * a20 - a10 * a22, i11 = a00 * a22 - a02 * a20, i12 = a02 * a10 - a00 * a12;
    const float i20 = a10 * a21 - a11 * a20, i21 = a01 * a20 - a00 * a21, i22 = a00 * a11 - a01 * a10;
    const float det = a00 * i00 + a10 * i01 + a20 * i02;
    const float J = det * E.invDetJ0[l];                                                                 // Eq. (12)
    const float P00 = E.P[0][l], P11 = E.P[1][l], P22 = E.P[2][l], P01 = E.P[3][l], P02 = E.P[4][l], P12 = E.P[5][l];
    const float PG = P00 * g11 + P11 * g22 + P22 * g33 + 2.f * (P01 * g12 + P02 * g13 + P12 * g23);        // mu Vol I1, Eq. (27)
    float y; int bad = 0;
    if (EXACT) { y = powf(J, -0.6666667f); bad = (J > 0.f) ? 0 : 2; }
    else
    {
        y = y_cache[l];
        float r = J * J * y * y * y;
        bad = !(fabsf(r - 1.f) < J23_TOL) | !(J > 0.f);
        for (int it = 0; it < J23_ITERS; it++) { y = y * (4.f - r) * (1.f / 3.f); r = J * J * y * y * y; }
    }
    y_cache[l] = y;
    const float c = (-(1.f / 3.f) * y * PG + E.KV[l] * J * (J - 1.f)) / det;   // const1 / det: const1 * invJt = c * adj(Jt)
    f[3*W + l]  = y * (a00 * P00 + a10 * P01 + a20 * P02) + c * i00;
    f[4*W + l]  = y * (a01 * P00 + a11 * P01 + a21 * P02) + c * i10;
    f[5*W + l]  = y * (a02 * P00 + a12 * P01 + a22 * P02) + c * i20;
    f[6*W + l]  = y * (a00 * P01 + a10 * P11 + a20 * P12) + c * i01;
    f[7*W + l]  = y * (a01 * P01 + a11 * P11 + a21 * P12) + c * i11;
    f[8*W + l]  = y * (a02 * P01 + a12 * P11 + a22 * P12) + c * i21;
    f[9*W + l]  = y * (a00 * P02 + a10 * P12 + a20 * P22) + c * i02;
    f[10*W + l] = y * (a01 * P02 + a11 * P12 + a21 * P22) + c * i12;
    f[11*W + l] = y * (a02 * P02 + a12 * P12 + a22 * P22) + c * i22;
    f[0*W + l] = -f[3*W + l] - f[6*W + l] - f[9*W + l];
    f[1*W + l] = -f[4*W + l] - f[7*W + l] - f[10*W + l];
    f[2*W + l] = -f[5*W + l] - f[8*W + l] - f[11*W + l];
    return bad;
}

// Forces of one block of W elements, stored in F (a flat array of FSTRIDE floats). U: displacements addressed by E.n.
// Returns the number of exact J^(-2/3) fall-backs; sets inverted if an element of the block has J <= 0.
static inline int blockForces(const EBlock& __restrict E, const float* __restrict U, float* __restrict y_cache, float* __restrict F, int& inverted)
{
    float u[12][W];
    for (int a = 0; a < 4; a++) { for (int l = 0; l < W; l++) { const float* p = U + E.n[a][l]; u[3 * a][l] = p[0]; u[3 * a + 1][l] = p[1]; u[3 * a + 2][l] = p[2]; } }
#if FAST_POWF
    // ablation: always use powf (no warm-started iteration), so there are no fall-backs to count
    int bad[W];
    for (int l = 0; l < W; l++) { bad[l] = laneForces<1>(E, l, u, y_cache, F); if (bad[l] == 2) { inverted = 1; } }
    return 0;
#else
    int bad[W], any_bad = 0;
#pragma omp simd reduction(|:any_bad)
    for (int l = 0; l < W; l++) { bad[l] = laneForces<0>(E, l, u, y_cache, F); any_bad |= bad[l]; }
    int n = 0;
    if (any_bad) { for (int l = 0; l < W; l++) { if (bad[l]) { if (laneForces<1>(E, l, u, y_cache, F) == 2) { inverted = 1; } n++; } } }   // rare
    return n;
#endif
}

// Explicit central-difference update of node i with internal force f. Returns true if a NaN appeared.
static FORCE_INLINE bool dofUpdate(const Solver& s, const size_t i, const float f[3], const float nfac,
                                   const float* __restrict Uc, const float* __restrict Up, float* __restrict Un)
{
    bool nan = false;
    for (size_t j = 0; j < 3; j++)
    {
        const size_t d = 3 * i + j;
        switch (s.kind[d])
        {
        case DOF_DISP:  Un[d] = s.dmag[d] * nfac; break;   // apply BC:Disp
        case DOF_FIXED: Un[d] = 0.f; break;                // apply BC:FixP
        default:                                           // explicit central-difference integration
            Un[d] = s.c1[d] * (s.fext[d] - f[j]) + s.c2[d] * Uc[d] + s.c3[d] * Up[d];
            nan |= std::isnan(Un[d]);
        }
    }
    return nan;
}

// sum the element forces of node i, then update it
static inline bool nodeUpdate(const Solver& s, const size_t i, const float nfac, const float* __restrict Fe,
                              const float* __restrict Uc, const float* __restrict Up, float* __restrict Un)
{
    float f[3] = { 0.f, 0.f, 0.f };
    for (uint32_t k = s.nptr[i]; k < s.nptr[i + 1]; k++) { const float* p = Fe + s.noff[k]; f[0] += p[0]; f[1] += p[W]; f[2] += p[2 * W]; }
    return dofUpdate(s, i, f, nfac, Uc, Up, Un);
}

/*------------------------------------------------------------------------------------------------- time loop */
// One explicit step, called by every thread of a parallel region (orphaned work-sharing loops, 2 barriers).
// Un = new displacements from Uc (current) and Up (previous). Adds STOP_INVERTED / STOP_NAN to *stop (this step's flag).
static void step(Solver& s, const float nfac, const float* __restrict Uc, const float* __restrict Up, float* __restrict Un,
                 long long& fallbacks, int* stop)
{
    // element pass: forces of every element block, then node pass: gather and update
#pragma omp for schedule(static)
    for (long b = 0; b < (long)s.nb; b++)
    {
        int inverted = 0;
        fallbacks += blockForces(s.blocks[b], Uc, &s.J23[(size_t)b * W], &s.Fe[(size_t)b * FSTRIDE], inverted);
        if (inverted)
        {
#pragma omp atomic
            *stop |= STOP_INVERTED;
        }
    }
    const float* Fe = s.Fe.data();
#pragma omp for schedule(static)
    for (long i = 0; i < (long)s.nn; i++)
    {
        if (nodeUpdate(s, (size_t)i, nfac, Fe, Uc, Up, Un))
        {
#pragma omp atomic
            *stop |= STOP_NAN;
        }
    }
}

// Validate the final displacement field before it is accepted and exported: every value finite, no element inverted.
static bool finalStateValid(const Solver& s)
{
    const float* U = s.U_curr;
    for (size_t i = 0; i < 3 * s.nn; i++) { if (!std::isfinite(U[i])) { return false; } }
    for (size_t b = 0; b < s.nb; b++)
    {
        const EBlock& E = s.blocks[b];
        for (int l = 0; l < W; l++)
        {
            const size_t e = b * W + (size_t)l;
            if (e >= s.ne) { continue; }   // padding lane
            const float u00 = U[E.n[0][l]], u01 = U[E.n[0][l] + 1], u02 = U[E.n[0][l] + 2];
            const float u10 = U[E.n[1][l]], u11 = U[E.n[1][l] + 1], u12 = U[E.n[1][l] + 2];
            const float u20 = U[E.n[2][l]], u21 = U[E.n[2][l] + 1], u22 = U[E.n[2][l] + 2];
            const float u30 = U[E.n[3][l]], u31 = U[E.n[3][l] + 1], u32 = U[E.n[3][l] + 2];
            const float a00 = E.J0[0][l] - u00 + u10, a01 = E.J0[1][l] - u01 + u11, a02 = E.J0[2][l] - u02 + u12;
            const float a10 = E.J0[3][l] - u00 + u20, a11 = E.J0[4][l] - u01 + u21, a12 = E.J0[5][l] - u02 + u22;
            const float a20 = E.J0[6][l] - u00 + u30, a21 = E.J0[7][l] - u01 + u31, a22 = E.J0[8][l] - u02 + u32;
            const float i00 = a11 * a22 - a12 * a21, i01 = a02 * a21 - a01 * a22, i02 = a01 * a12 - a02 * a11;
            const float det = a00 * i00 + a10 * i01 + a20 * i02;
            if (!(det * E.invDetJ0[l] > 0.f)) { return false; }
        }
    }
    return true;
}

static string waitPolicy()   // the OpenMP wait policy of this run (OMP_WAIT_POLICY, which main() sets if FAST_SPIN); "default" if none
{
    const char* value = getenv("OMP_WAIT_POLICY");
    string policy = (value != nullptr && *value != '\0') ? value : "default";
    transform(policy.begin(), policy.end(), policy.begin(), [](unsigned char c) { return (char)tolower(c); });
    return policy;
}

static bool runSimulation(const Model& m, Solver& s, VtkExporter& exporter)
{
    const int nthreads = omp_get_max_threads();
    int actual = 1;   // the team OpenMP actually creates (may be less than nthreads under OMP_THREAD_LIMIT)
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp single
        actual = omp_get_num_threads();
    }
    exporter.beginAnimation(m.num_steps, m.dt);   // does nothing unless the animation is switched on
    if (!exporter.ok()) { cerr << "\n\tError: animation output could not be started." << endl; return false; }
    auto nfacOf = [&](size_t stp) { return min(1.f, (stp + 1) * m.dt / m.total_t); };   // BC:Disp ramp, never above the full displacement
    cout << "\n\tusing " << actual << " threads (wait policy: " << waitPolicy() << ")" << endl;
    cout << "\tcomputing..." << endl;
    // Computation time = the solution only: the time loop's wall time, minus the time spent writing animation frames
    // (measured inside the exporter) and minus the time spent printing the progress lines (print_t, thread 0 only).
    // Reading the model, the set-up (reordering, precomputation) and the final-state export are outside the loop.
    chrono::high_resolution_clock::duration print_t(0);
    auto start_t = chrono::high_resolution_clock::now();

    float* Ubuf[3] = { s.U[0].data(), s.U[1].data(), s.U[2].data() };   // previous, current, next
    int stop[2] = { 0, 0 };   // STOP_NAN / STOP_INVERTED; even steps use stop[0], odd steps stop[1] (see the read below)
    long long fallbacks = 0;
    size_t steps_done = 0;
#if FAST_PERSTEP
    // ablation: one parallel region per step instead of one team for the whole time loop
    {
        float *Up = Ubuf[0], *Uc = Ubuf[1], *Un = Ubuf[2];
        size_t progress = 0, stp = 0;
        for (; stp < m.num_steps; stp++)
        {
            long long step_fallbacks = 0;
#pragma omp parallel num_threads(nthreads) reduction(+:step_fallbacks)
            {
                step(s, nfacOf(stp), Uc, Up, Un, step_fallbacks, &stop[stp & 1]);
            }
            fallbacks += step_fallbacks;
            float* t = Up; Up = Uc; Uc = Un; Un = t;
            if (stop[stp & 1]) { break; }
            if (exporter.isFrameStep(stp)) { exporter.writeFrame(stp, Uc); }
            if ((float)(stp + 1) / (float)m.num_steps * 100.f >= progress + 10)
            {
                const auto print_start = chrono::high_resolution_clock::now();
                progress += 10; cout << "\t\t\t(" << progress << "%)" << endl;
                print_t += chrono::high_resolution_clock::now() - print_start;
            }
        }
        s.U_curr = Uc; steps_done = stp + 1;
    }
#else
#pragma omp parallel num_threads(nthreads) reduction(+:fallbacks)   // one thread team for the whole time loop
    {
        float *Up = Ubuf[0], *Uc = Ubuf[1], *Un = Ubuf[2];   // every thread rotates its own copy: no shared write
        size_t progress = 0, stp = 0;
        for (; stp < m.num_steps; stp++)
        {
            step(s, nfacOf(stp), Uc, Up, Un, fallbacks, &stop[stp & 1]);
            float* t = Up; Up = Uc; Uc = Un; Un = t;
            // Every thread must read the same value here: if only some left the loop, the others would wait at the next
            // barrier for ever. A thread that is ahead may already be in step stp + 1 and set that step's flag, which is
            // the other one; it cannot get to step stp + 2, and to this flag again, before every thread has passed here.
            int st;
#pragma omp atomic read
            st = stop[stp & 1];
            if (st) { break; }
            if (exporter.isFrameStep(stp))
            {
#pragma omp single
                exporter.writeFrame(stp, Uc);
            }
            if (omp_get_thread_num() == 0 && (float)(stp + 1) / (float)m.num_steps * 100.f >= progress + 10)
            {
                const auto print_start = chrono::high_resolution_clock::now();
                progress += 10; cout << "\t\t\t(" << progress << "%)" << endl;
                print_t += chrono::high_resolution_clock::now() - print_start;   // only thread 0 writes print_t
            }
        }
        if (omp_get_thread_num() == 0) { s.U_curr = Uc; steps_done = stp + 1; }
    }
#endif
    const auto elapsed = chrono::high_resolution_clock::now() - start_t - exporter.animationTime() - print_t;   // solution only
    s.fallbacks = fallbacks;
    const int stopped = stop[0] | stop[1];
    if (!stopped && !finalStateValid(s))
    {
        cerr << "\n\tError: the final state contains a non-finite displacement or an inverted element; simulation aborted." << endl;
        exporter.endAnimation(false);
        return false;
    }
    exporter.endAnimation(!stopped);
    if (stopped & STOP_INVERTED) { cerr << "\n\tError: an element inverted (its Jacobian determinant became zero or negative) at step " << steps_done << ", simulation aborted. Try a smaller time step." << endl; return false; }
    if (stopped) { cerr << "\n\tError: solution diverged at step " << steps_done << ", simulation aborted. Try a smaller time step." << endl; return false; }
    cout << "\n\tComputation time:\t" << chrono::duration_cast<chrono::milliseconds>(elapsed).count() << " ms (solution only)" << endl;
    if (exporter.animation()) { cout << "\tAnimation export time:\t" << chrono::duration_cast<chrono::milliseconds>(exporter.animationTime()).count() << " ms (animation frames, not included above)" << endl; }
#if !FAST_POWF   // with FAST_POWF every element update is exact, so there is no fall-back count to report
    cout << "\tJ^(-2/3) exact fall-backs:\t" << s.fallbacks << " of " << (long long)m.num_steps * (long long)s.ne << " element updates" << endl;
#endif
    return true;
}

static void printInfo(const Model& m, const bool reorder)
{
    cout << endl;
    cout << "\t---------------------------------------------------------------------------------------------------" << endl;
    cout << "\t| DJ-TLED fast: optimised open-source (OpenMP + SIMD) implementation of:                          |" << endl;
    cout << "\t|               <A direct Jacobian total Lagrangian explicit dynamics finite element algorithm... |" << endl;
    cout << "\t|                                             for real-time simulation of hyperelastic materials. |" << endl;
    cout << "\t|                                                                              Zhang, J. (2021).  |" << endl;
    cout << "\t|                International Journal for Numerical Methods in Engineering, 122(20), 5744-5772.  |" << endl;
    cout << "\t|                                                                           doi:10.1002/nme.6772> |" << endl;
    cout << "\t|                                                            original DJ-TLED code by Jinao Zhang |" << endl;
    cout << "\t|                                                  optimised version (DJTLED_fast) by Baiqi Cheng |" << endl;
    cout << "\t---------------------------------------------------------------------------------------------------" << endl;
    cout << "\tModel:\t\t"      << m.fname << endl;
    cout << "\tNodes:\t\t"      << m.nodes() << " (" << 3 * m.nodes() << " DOFs)" << endl;
    cout << "\tElements:\t"     << m.tets.size() << " (" << m.ele_type << ")" << endl;
    cout << "\tEleMaterial:\t"  << m.material_type << ": " << m.mu << " " << m.K << "; Density: " << m.rho << endl;
    cout << "\tBC:\t\t"         << m.num_BCs << endl;
    cout << "\tDampingCoef.:\t" << m.alpha << endl;
    cout << "\tTimeStep:\t"     << m.dt << endl;
    cout << "\tTotalTime:\t"    << m.total_t << endl;
    cout << "\tNumSteps:\t"     << m.num_steps << endl;
    cout << "\tSpeed-ups:\t" << W << "-wide element blocks, "   // what this build and run really use (the ablation switches change it)
         << (FAST_POWF ? "powf for J^(-2/3) (ablation)" : "J^(-2/3) warm start") << ", "
         << (FAST_PERSTEP ? "a parallel region per step (ablation)" : "persistent thread team")
         << ", RCM reordering " << (reorder ? "on" : "off") << endl;
    cout << "\n\tNode index starts at " << m.node_begin_index << "." << endl;
    cout << "  \tElem index starts at " << m.ele_begin_index  << "." << endl;
}

int main(int argc, char **argv)
{
#if FAST_SPIN
    // waiting threads spin instead of sleeping between the two barriers of every step (must be set before the OpenMP
    // runtime starts; an OMP_WAIT_POLICY set by the user wins)
    setenv("OMP_WAIT_POLICY", "active", 0);
#endif
#if FAST_PCORES && defined(__APPLE__)
    // Apple silicon: by default use one thread per performance core. The efficiency cores are ~3x slower, and with
    // equal shares per thread every step would wait for them. OMP_NUM_THREADS set by the user wins.
    if (getenv("OMP_NUM_THREADS") == nullptr)
    {
        int pcores = 0; size_t len = sizeof(pcores);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &len, nullptr, 0) == 0 && pcores > 0) { omp_set_num_threads(pcores); }
    }
#endif
    const char* fname = nullptr; bool reorder = true; int threads = 0; long steps_cap = 0;
    VtkExportMode export_mode = VtkExportMode::Both;
    const char* usage = "Usage: djtled_fast <input.txt> [--export both|animation|final|none] [--threads N] [--steps N] [--no-reorder]";
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--export") == 0 && i + 1 < argc && parseVtkExportMode(argv[i + 1], export_mode)) { i++; }
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
        {
            char* end = nullptr;
            const long v = strtol(argv[i + 1], &end, 10);
            if (end != argv[i + 1] && *end == '\0' && v > 0) { threads = (int)v; i++; }
            else { cerr << "\n\tError: --threads needs a whole number of 1 or more. " << usage << endl; return EXIT_FAILURE; }
        }
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
        {
            char* end = nullptr;
            const long v = strtol(argv[i + 1], &end, 10);
            if (end != argv[i + 1] && *end == '\0' && v >= 1) { steps_cap = v; i++; }
            else { cerr << "\n\tError: --steps needs a whole number of 1 or more. " << usage << endl; return EXIT_FAILURE; }
        }
        else if (strcmp(argv[i], "--no-reorder") == 0) { reorder = false; }
        else if (strncmp(argv[i], "--", 2) == 0) { cerr << "\n\tError: unknown option or missing value: " << argv[i] << ". " << usage << endl; return EXIT_FAILURE; }
        else if (fname == nullptr) { fname = argv[i]; }
        else { cerr << "\n\tError: more than one input file given (" << fname << " and " << argv[i] << "). " << usage << endl; return EXIT_FAILURE; }
    }
    if (fname == nullptr) { cerr << "\n\tError: missing input argument (e.g., NH.txt). " << usage << endl; return EXIT_FAILURE; }
    if (threads > 0) { omp_set_num_threads(threads); }   // --threads wins over the defaults above
    Model m;
    const auto read_start = chrono::high_resolution_clock::now();
    if (!readModel(fname, m)) { return EXIT_FAILURE; }
    string out_name = SOLVER_NAME;
    if (steps_cap > 0 && (size_t)steps_cap < m.num_steps)
    {
        m.num_steps = (size_t)steps_cap;   // partial run: same ramp, fewer steps
        out_name += "_steps" + to_string((size_t)steps_cap);   // mark the output so it does not overwrite the full run
    }
    auto preparation = chrono::high_resolution_clock::now() - read_start;   // reading the model and set-up; reported, not part of the computation time
    printInfo(m, reorder);
    Solver s;
    const auto setup_start = chrono::high_resolution_clock::now();
    if (!setup(m, s, reorder)) { return EXIT_FAILURE; }
    preparation += chrono::high_resolution_clock::now() - setup_start;
    cout << "\n\tPreparation time:\t" << chrono::duration_cast<chrono::milliseconds>(preparation).count() << " ms (reading the model and set-up; not part of the computation time)" << endl;
    // VTK output (vtk_export.cpp): a genuinely flat copy of the element nodes. m.tets is a vector<array>, whose storage
    // is contiguous in practice but is not a single array object, so it cannot be indexed as one flat array by the exporter.
    vector<unsigned int> vtk_tets;
    if (export_mode != VtkExportMode::None)
    {
        vtk_tets.reserve(m.tets.size() * 4);
        for (const auto& t : m.tets) { for (unsigned int n : t) { vtk_tets.push_back(n); } }
    }
    VtkExporter exporter(out_name, m.fname, export_mode, VtkMesh{ m.nodes(), m.X.data(), m.tets.size(), vtk_tets.data(), s.newOfOld.data() });
    if (!runSimulation(m, s, exporter)) { return EXIT_FAILURE; }
    exporter.writeFinalState(s.U_curr);
    return exporter.ok() ? EXIT_SUCCESS : EXIT_FAILURE;   // fails if the final state or an animation frame could not be written
}
