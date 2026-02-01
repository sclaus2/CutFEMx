#!/usr/bin/env python3
"""Generate a sphere STL file for the FMM distance demo."""
import numpy as np

def generate_icosphere(subdivisions=4):
    """Generate an icosphere by subdividing an icosahedron."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (normalized to unit sphere)
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)
    vertices /= np.linalg.norm(vertices[0])
    
    # Icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    
    # Subdivide
    for _ in range(subdivisions):
        vertices, faces = subdivide(vertices, faces)
    
    return vertices, faces

def subdivide(vertices, faces):
    """Subdivide each triangle into 4 smaller triangles."""
    edge_midpoints = {}
    new_vertices = list(vertices)
    new_faces = []
    
    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge_midpoints:
            mid = (vertices[i] + vertices[j]) / 2
            mid /= np.linalg.norm(mid)  # Project to sphere
            edge_midpoints[key] = len(new_vertices)
            new_vertices.append(mid)
        return edge_midpoints[key]
    
    for f in faces:
        a, b, c = f
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.extend([
            [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
        ])
    
    return np.array(new_vertices), np.array(new_faces)

def write_stl(filename, vertices, faces, center, radius):
    """Write STL file."""
    with open(filename, 'w') as f:
        f.write("solid icosphere\n")
        for face in faces:
            v0, v1, v2 = [vertices[i] * radius + center for i in face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal /= np.linalg.norm(normal)
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            for v in [v0, v1, v2]:
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid icosphere\n")

if __name__ == "__main__":
    center = np.array([0.5, 0.5, 0.5])
    radius = 0.3
    subdivisions = 4
    
    vertices, faces = generate_icosphere(subdivisions)
    write_stl("sphere.stl", vertices, faces, center, radius)
    
    print(f"Generated sphere.stl:")
    print(f"  Center: {center}")
    print(f"  Radius: {radius}")
    print(f"  Triangles: {len(faces)}")
