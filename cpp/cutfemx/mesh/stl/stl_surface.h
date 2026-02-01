#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>

namespace cutfemx::mesh {

template <typename Real>
struct TriSoup
{
  // Vertex coordinates: length = 3*nV, layout [x0,y0,z0, x1,y1,z1, ...]
  std::vector<Real> X;

  // Triangle connectivity: length = 3*nT, indices into X vertices
  std::vector<std::int32_t> tri;

  // Per-triangle normals (STL facet normals as stored): length = 3*nT
  std::vector<Real> N;

  // Global triangle ids: length = nT (stable across ranks)
  std::vector<std::int32_t> tri_gid;

  std::int32_t nV() const { return static_cast<std::int32_t>(X.size()/3); }
  std::int32_t nT() const { return static_cast<std::int32_t>(tri.size()/3); }
};


/// Write TriSoup as VTK PolyData file
template <typename Real>
void write_trisoup_vtk(const TriSoup<Real>& soup, 
                       const std::string& filename)
{
    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "TriSoup surface\n";
    out << "ASCII\n";
    out << "DATASET POLYDATA\n";
    
    int nV = soup.nV();
    int nT = soup.nT();
    
    out << "POINTS " << nV << " double\n";
    for (int i = 0; i < nV; ++i)
    {
        out << soup.X[3*i] << " " << soup.X[3*i+1] << " " << soup.X[3*i+2] << "\n";
    }
    
    out << "POLYGONS " << nT << " " << 4*nT << "\n";
    for (int t = 0; t < nT; ++t)
    {
        out << "3 " << soup.tri[3*t] << " " << soup.tri[3*t+1] << " " << soup.tri[3*t+2] << "\n";
    }
    
    out.close();
    spdlog::info("Wrote surface: {} ({} triangles)", filename, nT);
}

/// Compute axis-aligned bounding box of a TriSoup surface
/// @param soup The triangular surface mesh
/// @return Pair of (min_corner, max_corner), each as array<Real, 3>
template <typename Real>
std::pair<std::array<Real, 3>, std::array<Real, 3>> 
bounding_box(const TriSoup<Real>& soup)
{
    std::array<Real, 3> min_corner = {
        std::numeric_limits<Real>::max(),
        std::numeric_limits<Real>::max(),
        std::numeric_limits<Real>::max()
    };
    std::array<Real, 3> max_corner = {
        std::numeric_limits<Real>::lowest(),
        std::numeric_limits<Real>::lowest(),
        std::numeric_limits<Real>::lowest()
    };
    
    const int nV = soup.nV();
    for (int i = 0; i < nV; ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            Real val = soup.X[3*i + d];
            if (val < min_corner[d]) min_corner[d] = val;
            if (val > max_corner[d]) max_corner[d] = val;
        }
    }
    
    return {min_corner, max_corner};
}

} // namespace cutfemx::mesh
