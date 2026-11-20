// MIT License. Copyright (c) 2022 Jinao Zhang, Copyright (c) 2026 Baiqi Cheng. Full text: LICENSE.
/*
vtk_export.h / vtk_export.cpp: VTK output shared by DJTLED.cpp and DJTLED_fast.cpp.

Two kinds of output, chosen with VtkExportMode (command line: --export both|animation|final|none):
  animation    results/<prefix>_animation.xdmf (open this in ParaView, with the "XDMF Reader") and its data in
               results/<prefix>_timesteps/: topology.bin (the elements, written once and shared by every frame) and,
               per frame, step_<step>_points.bin (deformed coordinates) and step_<step>_displacement.bin; about 100
               frames. The .bin files are raw 32-bit values in the byte order of the machine that wrote them.
  final state  results/<prefix>_deformed.vtk and results/<prefix>_undeformed.vtk (legacy VTK, ASCII)
<prefix> is "<algorithm name>_<model name>", the model name being the input file name without its folder and
extension, e.g. DJTLED_NH_001_tension. The folder "results" is created in the folder the solver is run from.

Use:
    VtkExporter exporter("DJTLED", model_path, mode, mesh);
    exporter.beginAnimation(num_steps, dt);
    for every step:  if (exporter.isFrameStep(step)) { exporter.writeFrame(step, U); }
    exporter.endAnimation(true);
    exporter.writeFinalState(U);
    if (!exporter.ok()) { ... a file could not be written ... }
Calls for a kind of output that is switched off do nothing, so the solver does not need to check the mode.
*/
#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

enum class VtkExportMode { None = 0, Final = 1, Animation = 2, Both = 3 };

// "none", "final", "animation" or "both" -> mode. Returns false (mode unchanged) for anything else.
bool parseVtkExportMode(const std::string& text, VtkExportMode& mode);

// A view of the solver's mesh; nothing is copied, so the arrays must outlive the exporter.
struct VtkMesh
{
    size_t              num_nodes;
    const float*        X;          // node coordinates, 3 per node
    size_t              num_tets;
    const unsigned int* tets;       // node indices (from 0), 4 per tetrahedron
    const uint32_t*     node_map;   // displacement of node i is U[3 * node_map[i] .. + 2]; nullptr: U[3 * i .. + 2]
};

class VtkExporter
{
public:
    typedef std::chrono::high_resolution_clock Clock;

    VtkExporter(const std::string& algorithm, const std::string& model_path, VtkExportMode mode, const VtkMesh& mesh);

    bool animation()  const { return (static_cast<int>(m_mode) & static_cast<int>(VtkExportMode::Animation)) != 0; }
    bool finalState() const { return (static_cast<int>(m_mode) & static_cast<int>(VtkExportMode::Final)) != 0; }
    bool ok()         const { return !m_failed; }   // false once any file could not be written

    // Animation. A frame is written every num_steps / 100 steps and after the last step; the frame written after
    // step i (counted from 0) is the state at time (i + 1) * dt.
    void beginAnimation(size_t num_steps, float dt);     // creates the frame folder, writes the elements and starts the .xdmf file
    // isFrameStep depends on the step alone, so every thread of a parallel region gets the same answer for a step (it
    // must not look at m_failed, which writeFrame may be changing on another thread at that moment).
    bool isFrameStep(size_t step) const { return m_frames_open && (step % m_interval == 0 || step + 1 == m_num_steps); }
    bool writeFrame(size_t step, const float* U);        // one thread at a time; after a write error it does nothing
    void endAnimation(bool completed);                   // finishes the .xdmf file
    Clock::duration animationTime() const { return m_animation_time; }   // time spent in writeFrame, for the solver's timer

    // Final state: the deformed and the undeformed mesh, both with the displacement field. False on a write error.
    bool writeFinalState(const float* U);

private:
    VtkExportMode   m_mode;
    VtkMesh         m_mesh;
    std::string     m_dir;                               // "results/"
    std::string     m_prefix, m_frames_dir, m_xdmf_name; // names inside m_dir
    std::ofstream   m_xdmf;
    std::vector<float> m_buffer;                         // one frame's coordinates and displacements
    bool            m_frames_open, m_failed;
    size_t          m_num_steps, m_interval;
    float           m_dt;
    Clock::duration m_animation_time;
};

#endif
