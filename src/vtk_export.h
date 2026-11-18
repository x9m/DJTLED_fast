// MIT License. Copyright (c) 2022 Jinao Zhang, Copyright (c) 2026 Baiqi Cheng. Full text: LICENSE.
/*
vtk_export.h / vtk_export.cpp: VTK output shared by DJTLED.cpp and DJTLED_fast.cpp.

Two kinds of output, chosen with VtkExportMode (command line: --export both|animation|final|none):
  animation    results/<prefix>_animation.pvd (open this in ParaView) + results/<prefix>_timesteps/step_<step>.vtk,
               about 100 frames
  final state  results/<prefix>_deformed.vtk and results/<prefix>_undeformed.vtk
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
    void beginAnimation(size_t num_steps, float dt);     // creates the frame folder and starts the .pvd file
    bool isFrameStep(size_t step) const { return m_frames_open && !m_failed && (step % m_interval == 0 || step + 1 == m_num_steps); }
    bool writeFrame(size_t step, const float* U);        // one thread at a time; after a write error no more frames are written
    void endAnimation(bool completed);                   // finishes the .pvd file
    Clock::duration animationTime() const { return m_animation_time; }   // time spent in writeFrame, for the solver's timer

    // Final state: the deformed and the undeformed mesh, both with the displacement field. False on a write error.
    bool writeFinalState(const float* U);

private:
    VtkExportMode   m_mode;
    VtkMesh         m_mesh;
    std::string     m_dir;                               // "results/"
    std::string     m_prefix, m_frames_dir, m_pvd_name;  // names inside m_dir
    std::ofstream   m_pvd;
    bool            m_frames_open, m_failed;
    size_t          m_num_steps, m_interval;
    float           m_dt;
    Clock::duration m_animation_time;
};

#endif
