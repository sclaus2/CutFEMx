// Copyright (c) 2022 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <cmath>
#include <concepts>

#include <cutcells/cell_types.h>

namespace cutfemx::quadrature
{

/// Computation of geometric transformations for interval

/// Compute Jacobian J for interval
template <std::floating_point T>
inline void compute_jacobian_interval_1d(const T vertex_coords[2], T J[1])
{
  J[0] = vertex_coords[1] - vertex_coords[0];
}

/// Compute Jacobian J for interval embedded in 2D
template <std::floating_point T>
inline void compute_jacobian_interval_2d(const T vertex_coords[4], T J[2])
{
  J[0] = vertex_coords[2] - vertex_coords[0];
  J[1] = vertex_coords[3] - vertex_coords[1];
}

/// Compute Jacobian J for interval embedded in 3D
template <std::floating_point T>
inline void compute_jacobian_interval_3d(const T vertex_coords[6], T J[3])
{
  J[0] = vertex_coords[3] - vertex_coords[0];
  J[1] = vertex_coords[4] - vertex_coords[1];
  J[2] = vertex_coords[5] - vertex_coords[2];
}

/// Compute Jacobian determinant for interval
template <std::floating_point T>
inline void compute_jacobian_determinants_interval_1d(const T J[1], T & det)
{
  det = J[0];
}

/// Compute Jacobian (pseudo)determinants for interval embedded in 2D
template <std::floating_point T>
inline void compute_jacobian_determinants_interval_2d(const T J[2], T & det2, T & det)
{
  det2 = J[0]*J[0] + J[1]*J[1];
  det = std::sqrt(det2);
}

/// Compute Jacobian (pseudo)determinants for interval embedded in 3D
template <std::floating_point T>
inline void compute_jacobian_determinants_interval_3d(const T J[3], T & det2, T & det)
{
  det2 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2];
  det = std::sqrt(det2);
}

/// Compute Jacobian inverse K for interval
template <std::floating_point T>
inline void  compute_jacobian_inverse_interval_1d(const T& det, T K[1])
{
  K[0] = 1.0 / det;
}

/// Compute Jacobian (pseudo)inverse K for interval embedded in 2D
template <std::floating_point T>
inline void compute_jacobian_inverse_interval_2d(const T& det2, const T J[2], T K[2])
{
  K[0] = J[0] / det2;
  K[1] = J[1] / det2;
}

/// Compute Jacobian (pseudo)inverse K for interval embedded in 3D
template <std::floating_point T>
inline void compute_jacobian_inverse_interval_3d(const T& det2, const T J[3], T K[3])
{
  K[0] = J[0] / det2;
  K[1] = J[1] / det2;
  K[2] = J[2] / det2;
}

/// Computation of geometric transformations for triangle

/// Compute Jacobian J for triangle
template <std::floating_point T>
inline void compute_jacobian_triangle_2d(const T vertex_coords[6], T J[4])
{
  J[0] = vertex_coords[2] - vertex_coords[0];
  J[1] = vertex_coords[4] - vertex_coords[0];
  J[2] = vertex_coords[3] - vertex_coords[1];
  J[3] = vertex_coords[5] - vertex_coords[1];
}

/// Compute Jacobian J for triangle embedded in 3D
template <std::floating_point T>
inline void compute_jacobian_triangle_3d(const T vertex_coords[9], T J[6])
{
  J[0] = vertex_coords[3] - vertex_coords[0];
  J[1] = vertex_coords[6] - vertex_coords[0];
  J[2] = vertex_coords[4] - vertex_coords[1];
  J[3] = vertex_coords[7] - vertex_coords[1];
  J[4] = vertex_coords[5] - vertex_coords[2];
  J[5] = vertex_coords[8] - vertex_coords[2];
}

/// Compute Jacobian determinant for triangle
template <std::floating_point T>
inline void compute_jacobian_determinants_triangle_2d(const T J[4], T& det)
{
  det = J[0]*J[3] - J[1]*J[2];
}

/// Compute Jacobian (pseudo)determinants for triangle embedded in 3D
template <std::floating_point T>
inline void compute_jacobian_determinants_triangle_3d(const T J[6], T& den, T& det2, T c[3],
                                                      T& det)
{
  const T d_0 = J[2]*J[5] - J[4]*J[3];
  const T d_1 = J[4]*J[1] - J[0]*J[5];
  const T d_2 = J[0]*J[3] - J[2]*J[1];

  c[0] = J[0]*J[0] + J[2]*J[2] + J[4]*J[4];
  c[1] = J[1]*J[1] + J[3]*J[3] + J[5]*J[5];
  c[2] = J[0]*J[1] + J[2]*J[3] + J[4]*J[5];

  den = c[0]*c[1] - c[2]*c[2];

  det2 = d_0*d_0 + d_1*d_1 + d_2*d_2;
  det = std::sqrt(det2);
}

/// Compute Jacobian inverse K for triangle
template <std::floating_point T>
inline void compute_jacobian_inverse_triangle_2d(const T J[4], const T& det, T K[4])
{
  K[0] =  J[3] / det;
  K[1] = -J[1] / det;
  K[2] = -J[2] / det;
  K[3] =  J[0] / det;
}

/// Compute Jacobian (pseudo)inverse K for triangle embedded in 3D
template <std::floating_point T>
inline void compute_jacobian_inverse_triangle_3d(const T J[6], const T& den, const T c[3], T K[6])
{
  K[0] = (J[0]*c[1] - J[1]*c[2]) / den;
  K[1] = (J[2]*c[1] - J[3]*c[2]) / den;
  K[2] = (J[4]*c[1] - J[5]*c[2]) / den;
  K[3] = (J[1]*c[0] - J[0]*c[2]) / den;
  K[4] = (J[3]*c[0] - J[2]*c[2]) / den;
  K[5] = (J[5]*c[0] - J[4]*c[2]) / den;
}

/// Computation of geometric transformations for tetrahedron

/// Compute Jacobian J for tetrahedron
template <std::floating_point T>
inline void compute_jacobian_tetrahedron_3d(const T vertex_coords[12], T J[9])
{
  J[0] = vertex_coords[3]  - vertex_coords[0];
  J[1] = vertex_coords[6]  - vertex_coords[0];
  J[2] = vertex_coords[9]  - vertex_coords[0];
  J[3] = vertex_coords[4]  - vertex_coords[1];
  J[4] = vertex_coords[7]  - vertex_coords[1];
  J[5] = vertex_coords[10] - vertex_coords[1];
  J[6] = vertex_coords[5]  - vertex_coords[2];
  J[7] = vertex_coords[8]  - vertex_coords[2];
  J[8] = vertex_coords[11] - vertex_coords[2];
}

/// Compute Jacobian determinant for tetrahedron
template <std::floating_point T>
inline void compute_jacobian_determinants_tetrahedron_3d(const T J[9], T d[9], T & det)
{
  d[0*3 + 0] = J[4]*J[8] - J[5]*J[7];
  d[0*3 + 1] = J[5]*J[6] - J[3]*J[8];
  d[0*3 + 2] = J[3]*J[7] - J[4]*J[6];
  d[1*3 + 0] = J[2]*J[7] - J[1]*J[8];
  d[1*3 + 1] = J[0]*J[8] - J[2]*J[6];
  d[1*3 + 2] = J[1]*J[6] - J[0]*J[7];
  d[2*3 + 0] = J[1]*J[5] - J[2]*J[4];
  d[2*3 + 1] = J[2]*J[3] - J[0]*J[5];
  d[2*3 + 2] = J[0]*J[4] - J[1]*J[3];

  det = J[0]*d[0*3 + 0] + J[3]*d[1*3 + 0] + J[6]*d[2*3 + 0];
}

/// Compute Jacobian inverse K for tetrahedron
template <std::floating_point T>
inline void  compute_jacobian_inverse_tetrahedron_3d(const T& det, const T d[9], T K[9])
{
  K[0] = d[0*3 + 0] / det;
  K[1] = d[1*3 + 0] / det;
  K[2] = d[2*3 + 0] / det;
  K[3] = d[0*3 + 1] / det;
  K[4] = d[1*3 + 1] / det;
  K[5] = d[2*3 + 1] / det;
  K[6] = d[0*3 + 2] / det;
  K[7] = d[1*3 + 2] / det;
  K[8] = d[2*3 + 2] / det;
}


template <std::floating_point T>
void compute_jacobian_interval(const std::size_t& gdim, std::span<const T> vertex_coords, std::span<T> J)
{
  switch(gdim)
  {
    case 1: compute_jacobian_interval_1d<T>(vertex_coords.data(), J.data());
            break;
    case 2: compute_jacobian_interval_2d<T>(vertex_coords.data(), J.data());
            break;
    case 3: compute_jacobian_interval_3d<T>(vertex_coords.data(), J.data());
            break;
    default: throw std::invalid_argument("cannot compute mapping with this gdim");
      break;
  }
}

template <std::floating_point T>
void compute_determinant_and_inverse_interval(const std::size_t& gdim, std::span<const T>& J,
                                              std::vector<T>& K, T& detJ)
{
  switch(gdim)
  {
    case 1: {compute_jacobian_determinants_interval_1d<T>(J.data(), detJ);
            compute_jacobian_inverse_interval_1d<T>(detJ, K.data());
            break;}
    case 2: {T det2 = 0.0;
            compute_jacobian_determinants_interval_2d<T>(J.data(), det2, detJ);
            compute_jacobian_inverse_interval_2d<T>(det2, J.data(), K.data());
            break;}
    case 3: {T det2 = 0.0;
            compute_jacobian_determinants_interval_3d<T>(J.data(), det2, detJ);
            compute_jacobian_inverse_interval_3d<T>(det2, J.data(), K.data());
            break;}
    default: throw std::invalid_argument("cannot compute mapping with this gdim");
      break;
  }
}

template <std::floating_point T>
void compute_jacobian_determinant_interval(const std::size_t& gdim, std::span<const T>& J, T& detJ)
{
  T _detJ = 0.0;

  switch(gdim)
  {
    case 1: {compute_jacobian_determinants_interval_1d<T>(J.data(), _detJ);
            break;}
    case 2: {T det2 = 0.0;
            compute_jacobian_determinants_interval_2d<T>(J.data(), det2, _detJ);
            break;}
    case 3: {T det2 = 0.0;
            compute_jacobian_determinants_interval_3d<T>(J.data(), det2, _detJ);
            break;}
    default: throw std::invalid_argument("cannot compute mapping with this gdim");
      break;
  }
}

template <std::floating_point T>
void compute_jacobian_triangle(const std::size_t& gdim, std::span<const T> vertex_coords, std::span<T> J)
{
  switch(gdim)
  {
    case 2: compute_jacobian_triangle_2d<T>(vertex_coords.data(), J.data());
            break;
    case 3: compute_jacobian_triangle_3d<T>(vertex_coords.data(), J.data());
            break;
    default: throw std::invalid_argument("cannot compute mapping with this gdim");
      break;
  }
}

template <std::floating_point T>
void compute_determinant_and_inverse_triangle(const std::size_t& gdim, std::span<const T>& J,
                                              std::vector<T>& K, T& detJ)
{
  switch(gdim)
  {
    case 2: {compute_jacobian_determinants_triangle_2d<T>(J.data(), detJ);
            compute_jacobian_inverse_triangle_2d<T>(J.data(), detJ, K.data());
            break;}
    case 3: {T det2 = 0.0;
            T den = 0.0;
            T c[3];
            compute_jacobian_determinants_triangle_3d<T>(J.data(),den,det2,c,detJ);
            compute_jacobian_inverse_triangle_3d<T>(J.data(), den, c, K.data());
            break;}
    default: {throw std::invalid_argument("cannot compute mapping with this gdim");
      break;}
  }
}

template <std::floating_point T>
void compute_jacobian_determinant_triangle(const std::size_t& gdim, std::span<const T>& J, T& detJ)
{
  T _detJ = 0.0;

  switch(gdim)
  {
    case 2: {compute_jacobian_determinants_triangle_2d<T>(J.data(), _detJ);
            break;}
    case 3: {T det2 = 0.0;
            T den = 0.0;
            T c[3];
            compute_jacobian_determinants_triangle_3d<T>(J.data(),den,det2,c,_detJ);
            break;}
    default: {throw std::invalid_argument("cannot compute mapping with this gdim");
      break;}
  }
}


template <std::floating_point T>
void compute_jacobian_tetrahedron(const std::size_t& gdim, std::span<const T> vertex_coords, std::span<T> J)
{
  switch(gdim)
  {
    case 3: {compute_jacobian_tetrahedron_3d<T>(vertex_coords.data(), J.data());
            break;}
    default: {throw std::invalid_argument("cannot compute mapping with this gdim");
      break;}
  }
}

template <std::floating_point T>
void compute_jacobian_determinant_tetrahedron(const std::size_t& gdim, std::span<const T>& J, T& detJ)
{
  T _detJ = 0.0;

  switch(gdim)
  {
    case 3: {T d[9];
            compute_jacobian_determinants_tetrahedron_3d<T>(J.data(), d, detJ);
            break;}
    default: {throw std::invalid_argument("cannot compute mapping with this gdim");
      break;}
  }
}

template <std::floating_point T>
void compute_determinant_and_inverse_tetrahedron(const std::size_t& gdim, std::span<const T>& J,
                                              std::vector<T>& K, T& detJ)
{
  switch(gdim)
  {
    case 3: {T d[9];
            compute_jacobian_determinants_tetrahedron_3d<T>(J.data(), d, detJ);
            compute_jacobian_inverse_tetrahedron_3d<T>(detJ,d,K.data());
            break;}
    default: {throw std::invalid_argument("cannot compute mapping with this gdim");
      break;}
  }
}

template <std::floating_point T>
void compute_jacobian(const cutcells::cell::type &cell_type, const std::size_t& gdim,
                      std::span<const T> vertex_coords, std::vector<T>& J)
{
  int tdim = cutcells::cell::get_tdim(cell_type);
  assert(J.size() == tdim*gdim);

  std::fill(J.begin(), J.end(), 0.0);

  std::span<T> J_s(J);

  //check if vertex coordinates has correct size
  int num_vertices = cutcells::cell::get_num_vertices(cell_type);
  assert(vertex_coords.size() == gdim*num_vertices);

  switch(cell_type)
  {
    case cutcells::cell::type::interval: compute_jacobian_interval<T>(gdim, vertex_coords, J_s);
                        break;
    case cutcells::cell::type::triangle: compute_jacobian_triangle<T>(gdim, vertex_coords, J_s);
                        break;
    case cutcells::cell::type::tetrahedron: compute_jacobian_tetrahedron<T>(gdim, vertex_coords, J_s);
                        break;
    default: throw std::invalid_argument("computation of jacobians for this cell type is not implemented yet");
      break;
  }
}

template <std::floating_point T>
void compute_jacobian_determinant(const cutcells::cell::type &cell_type, const std::size_t& gdim,
                                              std::span<const T> vertex_coords, T& detJ)
{
  //compute Jacobian first
  int tdim = cutcells::cell::get_tdim(cell_type);
  std::vector<T> J(tdim*gdim);
  std::vector<T> K(gdim*tdim);
  compute_jacobian(cell_type, gdim, vertex_coords, J);

  std::span<T> J_s(J);

  switch(cell_type)
  {
    case cutcells::cell::type::point: detJ = 1.0;
    case cutcells::cell::type::interval: compute_jacobian_determinant_interval<T>(gdim, J_s, detJ);
                                         break;
    case cutcells::cell::type::triangle: compute_jacobian_determinant_triangle<T>(gdim, J_s, detJ);
                        break;
    case cutcells::cell::type::tetrahedron:compute_jacobian_determinant_tetrahedron<T>(gdim, J_s, detJ);
                        break;
    default: throw std::invalid_argument("computation of jacobians for this cell type is not implemented yet");
      break;
  }
}

// Create transformation matrices for mappings from physical cell to reference cell and back
// for affine intervals, triangles and tetrahedra
// i.e. compute the jacobian, the jacobian inverse and the determinante of the jacobian
// Note this function assumes that J and K have been allocated outside this function
// input: type of cell, geometric dimension, physical vertex coordinates of cell
// output: Jacobian, inverse Jacobian K and determinante of Jacobian
template <std::floating_point T>
void compute_jacobian_determinant_and_inverse(const cutcells::cell::type &cell_type, const std::size_t& gdim,
                                              std::span<const T> vertex_coords,
                                              std::vector<T>& J, std::vector<T>& K, T& detJ)
{
  //compute Jacobian first
    int tdim = cutcells::cell::get_tdim(cell_type);
  compute_jacobian(cell_type, gdim, vertex_coords, J);

  // check that K has expected size
  assert(K.size() == gdim*tdim);
  //set K to zero
  std::fill(K.begin(), K.end(), 0.0);
  //set detJ to zero
  detJ = 0.0;
  std::span<const T> J_s(J); 

  switch(cell_type)
  {
    case cutcells::cell::type::interval: {compute_determinant_and_inverse_interval<T>(gdim,J_s,K,detJ);
                                         break;}
    case cutcells::cell::type::triangle: {compute_determinant_and_inverse_triangle<T>(gdim,J_s,K,detJ);
                        break;}
    case cutcells::cell::type::tetrahedron: {compute_determinant_and_inverse_tetrahedron<T>(gdim,J_s,K,detJ);
                        break;}
    default:{ throw std::invalid_argument("computation of jacobians for this cell type is not implemented yet");
      break;}
  }
}

}