// MIT License. Copyright (c) 2022 Jinao Zhang, Copyright (c) 2026 Baiqi Cheng. Full text: LICENSE.
// VTK output shared by DJTLED.cpp and DJTLED_fast.cpp; see vtk_export.h.
#include "vtk_export.h"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sys/stat.h> // mkdir
using namespace std;

static_assert(sizeof(unsigned int) == 4 && sizeof(float) == 4, "the animation's binary files hold 32-bit integers and floats");

bool parseVtkExportMode(const string& text, VtkExportMode& mode)
{
    if      (text == "none")      { mode = VtkExportMode::None; }
    else if (text == "final")     { mode = VtkExportMode::Final; }
    else if (text == "animation") { mode = VtkExportMode::Animation; }
    else if (text == "both")      { mode = VtkExportMode::Both; }
    else { return false; }
    return true;
}

static const char* const RESULTS_DIR = "results";   // every output file goes into this folder (created in the current folder)

static void makeFolder(const string& path)          // no error if it already exists
{
    mkdir(path.c_str(), 0755);
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

// Escape a file name before it is written into XML (a model name with '&' or '<' would otherwise produce an .xdmf
// file that an XML parser rejects).
static string xmlEscape(const string& text)
{
    string out;
    out.reserve(text.size());
    for (char c : text)
    {
        if      (c == '&')  { out += "&amp;"; }
        else if (c == '<')  { out += "&lt;"; }
        else if (c == '>')  { out += "&gt;"; }
        else if (c == '"')  { out += "&quot;"; }
        else if (c == '\'') { out += "&apos;"; }
        else { out += c; }
    }
    return out;
}

// One legacy-VTK unstructured grid in ASCII: the mesh (deformed by U, or as it is in the model) and the displacement
// field. file: path to write; title: the VTK title line (the file name without the results folder).
static bool writeGrid(const string& file, const string& title, const VtkMesh& mesh, const float* U, const bool deformed)
{
    ofstream fout(file.c_str());
    if (!fout.is_open()) { cerr << "\n\tError: cannot open " << file << " for writing, results not saved." << endl; return false; }
    fout << setprecision(numeric_limits<float>::max_digits10);   // full float precision: the files round-trip exactly
    fout << "# vtk DataFile Version 3.0" << '\n';                // the legacy format with this CELLS layout (before 5.0)
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

// The animation's data files: count 32-bit values, written as they are in memory. False (with a message) on any error.
static bool writeRaw(const string& file, const void* data, const size_t count)
{
    ofstream fout(file.c_str(), ios::binary);
    if (fout.is_open()) { fout.write(static_cast<const char*>(data), static_cast<streamsize>(count * 4)); fout.flush(); }
    if (!fout) { cerr << "\n\tError: writing " << file << " failed (disk full?), animation not saved." << endl; return false; }
    return true;
}

// The XDMF description of one of those files: rows x columns values of the given type ("Int" or "Float"), in this
// machine's byte order. file is relative to the .xdmf file.
static string dataItem(const char* type, const size_t rows, const int columns, const string& file)
{
    const uint16_t one = 1;
    const bool little_endian = *reinterpret_cast<const unsigned char*>(&one) == 1;
    return string("<DataItem Format=\"Binary\" DataType=\"") + type + "\" Precision=\"4\" Endian=\"" + (little_endian ? "Little" : "Big")
         + "\" Dimensions=\"" + to_string(rows) + " " + to_string(columns) + "\">" + xmlEscape(file) + "</DataItem>";
}

VtkExporter::VtkExporter(const string& algorithm, const string& model_path, VtkExportMode mode, const VtkMesh& mesh) :
    m_mode(mode), m_mesh(mesh), m_dir(string(RESULTS_DIR) + "/"), m_prefix(algorithm + "_" + modelName(model_path)),
    m_frames_dir(m_prefix + "_timesteps"), m_xdmf_name(m_prefix + "_animation.xdmf"),
    m_frames_open(false), m_failed(false), m_num_steps(0), m_interval(1), m_dt(0.f), m_animation_time(0) {}

void VtkExporter::beginAnimation(size_t num_steps, float dt)
{
    if (!animation()) { return; }
    m_num_steps = num_steps; m_dt = dt;
    m_interval = max(size_t(1), num_steps / 100);
    makeFolder(RESULTS_DIR);
    makeFolder(m_dir + m_frames_dir);
    // the elements do not change during a run: they are written once, and every frame refers to this file
    if (!writeRaw(m_dir + m_frames_dir + "/topology.bin", m_mesh.tets, m_mesh.num_tets * 4)) { m_failed = true; }
    m_xdmf.open((m_dir + m_xdmf_name).c_str());   // the list of frames: their times and their data files
    if (m_xdmf.is_open()) { m_xdmf << "<?xml version=\"1.0\"?>\n<Xdmf Version=\"2.0\">\n  <Domain>\n    <Grid Name=\"animation\" GridType=\"Collection\" CollectionType=\"Temporal\">\n"; }
    else { cerr << "\n\tError: cannot open " << m_dir + m_xdmf_name << " for writing, animation not saved." << endl; m_failed = true; }
    m_frames_open = true;
    cout << "\tExporting every " << m_interval << " steps (total steps: " << num_steps << ")" << endl;
}

bool VtkExporter::writeFrame(size_t step, const float* U)
{
    if (!m_frames_open || m_failed) { return !m_failed; }
    const Clock::time_point start = Clock::now();
    char name[32];
    snprintf(name, sizeof(name), "step_%06zu", step);
    const string base = m_frames_dir + "/" + name;   // relative to the .xdmf file, which is in the same results folder
    // deformed coordinates, then displacements, both in the model's node order
    const size_t n = m_mesh.num_nodes;
    m_buffer.resize(6 * n);
    float* x = m_buffer.data();
    float* u = x + 3 * n;
    for (size_t i = 0; i < n; i++)
    {
        const float* ui = U + 3 * (m_mesh.node_map != nullptr ? (size_t)m_mesh.node_map[i] : i);
        for (int k = 0; k < 3; k++) { u[3 * i + k] = ui[k]; x[3 * i + k] = m_mesh.X[3 * i + k] + ui[k]; }
    }
    const bool ok = writeRaw(m_dir + base + "_points.bin", x, 3 * n) && writeRaw(m_dir + base + "_displacement.bin", u, 3 * n);
    if (!ok) { m_failed = true; }
    if (ok && m_xdmf.is_open())
    {
        m_xdmf << "      <Grid Name=\"" << name << "\" GridType=\"Uniform\">\n"
               << "        <Time Value=\"" << static_cast<float>(step + 1) * m_dt << "\"/>\n"
               << "        <Topology TopologyType=\"Tetrahedron\" NumberOfElements=\"" << m_mesh.num_tets << "\">" << dataItem("Int", m_mesh.num_tets, 4, m_frames_dir + "/topology.bin") << "</Topology>\n"
               << "        <Geometry GeometryType=\"XYZ\">" << dataItem("Float", n, 3, base + "_points.bin") << "</Geometry>\n"
               << "        <Attribute Name=\"displacement\" AttributeType=\"Vector\" Center=\"Node\">" << dataItem("Float", n, 3, base + "_displacement.bin") << "</Attribute>\n"
               << "      </Grid>\n";
    }
    if (!m_xdmf) { m_failed = true; }   // a full disk or a closed stream must not be reported as success
    m_animation_time += Clock::now() - start;
    return ok && !m_failed;
}

void VtkExporter::endAnimation(bool completed)
{
    if (!m_frames_open) { return; }
    m_frames_open = false;
    if (!m_xdmf.is_open()) { return; }
    m_xdmf << "    </Grid>\n  </Domain>\n</Xdmf>\n";
    m_xdmf.close();
    if (!m_xdmf) { m_failed = true; cerr << "\n\tError: writing or closing " << m_dir + m_xdmf_name << " failed, animation not saved." << endl; }
    if (completed && !m_failed) { cout << "\tCreated " << m_dir + m_xdmf_name << " - open this file in ParaView (reader: XDMF Reader) to view the complete animation" << endl; }
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
