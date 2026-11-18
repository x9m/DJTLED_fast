// MIT License. Copyright (c) 2022 Jinao Zhang, Copyright (c) 2026 Baiqi Cheng. Full text: LICENSE.
// VTK output shared by DJTLED.cpp and DJTLED_fast.cpp; see vtk_export.h.
#include "vtk_export.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#ifdef _WIN32
#include <direct.h>   // _mkdir
#else
#include <sys/stat.h> // mkdir
#endif
using namespace std;

bool parseVtkExportMode(const string& text, VtkExportMode& mode)
{
    if      (text == "none")      { mode = VtkExportMode::None; }
    else if (text == "final")     { mode = VtkExportMode::Final; }
    else if (text == "animation") { mode = VtkExportMode::Animation; }
    else if (text == "both")      { mode = VtkExportMode::Both; }
    else { return false; }
    return true;
}

static const char* const RESULTS_DIR = "results";   // every VTK file goes into this folder (created in the current folder)

static void makeFolder(const string& path)          // no error if it already exists
{
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

// "models/NH_001_tension.txt" -> "NH_001_tension"
static string modelName(const string& model_path)
{
    const size_t slash = model_path.find_last_of("/\\");
    string name = (slash == string::npos) ? model_path : model_path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    if (dot != string::npos && dot > 0) { name = name.substr(0, dot); }
    return name;
}

// One legacy-VTK unstructured grid: the mesh (deformed by U, or as it is in the model) and the displacement field.
// file: path to write; title: the VTK title line (the file name without the results folder).
static bool writeGrid(const string& file, const string& title, const VtkMesh& mesh, const float* U, const bool deformed)
{
    ofstream fout(file.c_str());
    if (!fout.is_open()) { cerr << "\n\tError: cannot open " << file << " for writing, results not saved." << endl; return false; }
    fout << "# vtk DataFile Version 3.8" << '\n';
    fout << title << '\n';
    fout << "ASCII" << '\n';
    fout << "DATASET UNSTRUCTURED_GRID" << '\n';
    fout << "POINTS " << mesh.num_nodes << " float" << '\n';
    for (size_t i = 0; i < mesh.num_nodes; i++)
    {
        const float* x = mesh.X + 3 * i;
        const float* u = U + 3 * (mesh.node_map != nullptr ? (size_t)mesh.node_map[i] : i);
        if (deformed) { fout << x[0] + u[0] << " " << x[1] + u[1] << " " << x[2] + u[2] << '\n'; }
        else          { fout << x[0] << " " << x[1] << " " << x[2] << '\n'; }
    }
    fout << "CELLS " << mesh.num_tets << " " << mesh.num_tets * (4 + 1) << '\n';
    for (size_t e = 0; e < mesh.num_tets; e++)
    {
        const unsigned int* t = mesh.tets + 4 * e;
        fout << 4 << " " << t[0] << " " << t[1] << " " << t[2] << " " << t[3] << '\n';
    }
    fout << "CELL_TYPES " << mesh.num_tets << '\n';
    for (size_t e = 0; e < mesh.num_tets; e++) { fout << 10 << '\n'; }   // 10 = tetrahedron
    fout << "POINT_DATA " << mesh.num_nodes << '\n';
    fout << "VECTORS " << "displacement" << " float" << '\n';
    for (size_t i = 0; i < mesh.num_nodes; i++)
    {
        const float* u = U + 3 * (mesh.node_map != nullptr ? (size_t)mesh.node_map[i] : i);
        fout << u[0] << " " << u[1] << " " << u[2] << '\n';
    }
    fout.flush();
    if (!fout) { cerr << "\n\tError: writing " << file << " failed (disk full?), results not saved." << endl; return false; }
    return true;
}

VtkExporter::VtkExporter(const string& algorithm, const string& model_path, VtkExportMode mode, const VtkMesh& mesh) :
    m_mode(mode), m_mesh(mesh), m_dir(string(RESULTS_DIR) + "/"), m_prefix(algorithm + "_" + modelName(model_path)),
    m_frames_dir(m_prefix + "_timesteps"), m_pvd_name(m_prefix + "_animation.pvd"),
    m_frames_open(false), m_failed(false), m_num_steps(0), m_interval(1), m_dt(0.f), m_animation_time(0) {}

void VtkExporter::beginAnimation(size_t num_steps, float dt)
{
    if (!animation()) { return; }
    m_num_steps = num_steps; m_dt = dt;
    m_interval = max(size_t(1), num_steps / 100);
    makeFolder(RESULTS_DIR);
    makeFolder(m_dir + m_frames_dir);
    m_pvd.open((m_dir + m_pvd_name).c_str());   // ParaView collection: the list of frames and their times
    if (m_pvd.is_open()) { m_pvd << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n  <Collection>\n"; }
    else { cerr << "\n\tError: cannot open " << m_dir + m_pvd_name << " for writing, animation not saved." << endl; m_failed = true; }
    m_frames_open = true;
    cout << "\tExporting every " << m_interval << " steps (total steps: " << num_steps << ")" << endl;
}

bool VtkExporter::writeFrame(size_t step, const float* U)
{
    if (!m_frames_open || m_failed) { return !m_failed; }
    const Clock::time_point start = Clock::now();
    char name[32];
    snprintf(name, sizeof(name), "/step_%06zu.vtk", step);
    const string file = m_frames_dir + name;   // relative to the .pvd file, which is in the same results folder
    const bool ok = writeGrid(m_dir + file, file, m_mesh, U, true);
    if (!ok) { m_failed = true; }
    if (ok && m_pvd.is_open()) { m_pvd << "    <DataSet timestep=\"" << static_cast<float>(step + 1) * m_dt << "\" file=\"" << file << "\"/>\n"; }
    m_animation_time += Clock::now() - start;
    return ok;
}

void VtkExporter::endAnimation(bool completed)
{
    if (!m_frames_open) { return; }
    m_frames_open = false;
    if (!m_pvd.is_open()) { return; }
    m_pvd << "  </Collection>\n</VTKFile>\n";
    m_pvd.close();
    if (completed && !m_failed) { cout << "\tCreated " << m_dir + m_pvd_name << " - open this file to view the complete animation" << endl; }
}

bool VtkExporter::writeFinalState(const float* U)
{
    if (!finalState()) { return true; }
    cout << "\texporting final results..." << endl;
    const string files[2] = { m_prefix + "_deformed.vtk", m_prefix + "_undeformed.vtk" };
    makeFolder(RESULTS_DIR);
    for (int k = 0; k < 2; k++)
    {
        if (!writeGrid(m_dir + files[k], files[k], m_mesh, U, k == 0)) { m_failed = true; return false; }
        cout << "\t\t\t" << m_dir + files[k] << endl;
    }
    cout << "\tVTK saved." << endl;
    return true;
}
