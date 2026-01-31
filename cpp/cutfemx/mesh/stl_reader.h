#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>

namespace cutfemx::mesh {

template <typename Real>
void read_stl_facets(const std::string& path,
                     std::vector<std::int32_t>& tri_gid,
                     std::vector<Real>& facet_data)
{
    // Check if binary or ASCII
    // open file
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open STL file: " + path);
    }

    // Read first 80 bytes
    char header[80];
    file.read(header, 80);
    
    // Check if "solid" starts the file -> ASCII, BUT valid binary stl can also start with "solid".
    // Better heuristic: check file size vs expected size from triangle count.
    
    bool is_binary = true;
    
    // Read triangle count
    std::uint32_t num_tris_binary = 0;
    file.read(reinterpret_cast<char*>(&num_tris_binary), 4);
    
    // Calculate expected size for binary
    // 80 + 4 + num_tris * 50
    // 50 bytes per triangle: 12 (normal) + 36 (3 vertices) + 2 (attr)
    file.seekg(0, std::ios::end);
    std::size_t file_size = file.tellg();
    
    std::size_t expected_size = 80 + 4 + num_tris_binary * 50;
    
    if (file_size != expected_size)
    {
        is_binary = false;
    }
    
    // Reset
    file.seekg(0, std::ios::beg);
    
    if (is_binary)
    {
        // Skip header
        file.seekg(84, std::ios::beg);
        
        tri_gid.reserve(num_tris_binary);
        facet_data.reserve(num_tris_binary * 12);
        
        // buffer for one triangle: 12 floats + 2 bytes
        // 12 * 4 + 2 = 50 bytes
        // But binary STL uses 32-bit float
        
        struct __attribute__((packed)) StlFacetBin {
            float n[3];
            float v0[3];
            float v1[3];
            float v2[3];
            std::uint16_t attr;
        };
        
        StlFacetBin facet;
        for (std::uint32_t i = 0; i < num_tris_binary; ++i)
        {
            file.read(reinterpret_cast<char*>(&facet), 50);
            
            tri_gid.push_back(i);
            
            // Convert to Real
            facet_data.push_back(static_cast<Real>(facet.n[0]));
            facet_data.push_back(static_cast<Real>(facet.n[1]));
            facet_data.push_back(static_cast<Real>(facet.n[2]));
            
            facet_data.push_back(static_cast<Real>(facet.v0[0]));
            facet_data.push_back(static_cast<Real>(facet.v0[1]));
            facet_data.push_back(static_cast<Real>(facet.v0[2]));
            
            facet_data.push_back(static_cast<Real>(facet.v1[0]));
            facet_data.push_back(static_cast<Real>(facet.v1[1]));
            facet_data.push_back(static_cast<Real>(facet.v1[2]));
            
            facet_data.push_back(static_cast<Real>(facet.v2[0]));
            facet_data.push_back(static_cast<Real>(facet.v2[1]));
            facet_data.push_back(static_cast<Real>(facet.v2[2]));
        }
    }
    else
    {
        // ASCII
        std::string line;
        std::string word;
        
        std::vector<Real> current_normal(3);
        std::vector<Real> current_vertices(9);
        int vertex_idx = 0;
        int t = 0;
        
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            ss >> word;
            if (word == "facet")
            {
                ss >> word; // normal
                if (word == "normal")
                {
                    double nx, ny, nz;
                    ss >> nx >> ny >> nz;
                    current_normal[0] = static_cast<Real>(nx);
                    current_normal[1] = static_cast<Real>(ny);
                    current_normal[2] = static_cast<Real>(nz);
                }
                vertex_idx = 0;
            }
            else if (word == "vertex")
            {
                if (vertex_idx < 3)
                {
                    double vx, vy, vz;
                    ss >> vx >> vy >> vz;
                    current_vertices[3*vertex_idx + 0] = static_cast<Real>(vx);
                    current_vertices[3*vertex_idx + 1] = static_cast<Real>(vy);
                    current_vertices[3*vertex_idx + 2] = static_cast<Real>(vz);
                    vertex_idx++;
                }
            }
            else if (word == "endfacet")
            {
                tri_gid.push_back(t++);
                
                // Push normal
                facet_data.push_back(current_normal[0]);
                facet_data.push_back(current_normal[1]);
                facet_data.push_back(current_normal[2]);
                
                // Push vertices
                for (int k=0; k<9; ++k)
                    facet_data.push_back(current_vertices[k]);
            }
        }
    }
}

} // namespace cutfemx::mesh
