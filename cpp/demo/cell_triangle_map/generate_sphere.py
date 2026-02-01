import numpy as np
import struct

def create_sphere(radius=0.35, center=(0.5, 0.5, 0.5), subdivisions=3):
    """
    Create a sphere using an icosahedron and subdivision.
    Returns vertices and faces.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Icosahedron vertices
    verts = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ]
    
    verts = np.array(verts)
    # Normalize
    verts = verts / np.linalg.norm(verts, axis=1)[:, None]
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    faces = np.array(faces)

    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache = {}

        def get_midpoint(i1, i2):
            nonlocal verts
            key = tuple(sorted((i1, i2)))
            if key in midpoint_cache:
                return midpoint_cache[key]
            
            mid = (verts[i1] + verts[i2]) / 2.0
            mid /= np.linalg.norm(mid)
            
            idx = len(verts)
            verts = np.vstack([verts, mid])
            midpoint_cache[key] = idx
            return idx

        for f in faces:
            a = get_midpoint(f[0], f[1])
            b = get_midpoint(f[1], f[2])
            c = get_midpoint(f[2], f[0])
            
            new_faces.append([f[0], a, c])
            new_faces.append([f[1], b, a])
            new_faces.append([f[2], c, b])
            new_faces.append([a, b, c])
            
        faces = np.array(new_faces)

    # Scale and translate
    verts = verts * radius + np.array(center)
    
    return verts, faces

def write_stl(filename, verts, faces):
    """
    Write ASCII STL file.
    """
    with open(filename, 'w') as f:
        f.write("solid sphere\n")
        
        for face in faces:
            v0 = verts[face[0]]
            v1 = verts[face[1]]
            v2 = verts[face[2]]
            
            # Compute normal
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm > 0:
                n /= norm
            
            f.write(f"facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("outer loop\n")
            f.write(f"vertex {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write("endloop\n")
            f.write("endfacet\n")
            
        f.write("endsolid sphere\n")

if __name__ == "__main__":
    verts, faces = create_sphere()
    filename = "sphere.stl"
    write_stl(filename, verts, faces)
    print(f"Saved {filename}")
