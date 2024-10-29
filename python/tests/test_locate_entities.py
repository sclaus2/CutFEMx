# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import pytest
from mpi4py import MPI
import numpy as np

from cutfemx.level_set import locate_entities, locate_entities_part
from dolfinx import fem, mesh

def test_locate_entities():
  N = 3

  msh = mesh.create_rectangle(
      comm=MPI.COMM_WORLD,
      points=((0.0, 0.0), (1.0, 1.0)),
      n=(N, N),
      cell_type=mesh.CellType.triangle,
  )

  def line(x):
    return x[0]-0.51

  V = fem.functionspace(msh, ("Lagrange", 1))

  level_set = fem.Function(V)
  level_set.interpolate(line)
  dim = msh.topology.dim

  intersected_entities = locate_entities(level_set,dim,"phi=0")

  entities_exact = np.array([2,4,7,10,13,15],dtype=np.int32)

  assert np.array_equal(intersected_entities,entities_exact)

def test_locate_entities_part():
  N = 3

  msh = mesh.create_rectangle(
      comm=MPI.COMM_WORLD,
      points=((0.0, 0.0), (1.0, 1.0)),
      n=(N, N),
      cell_type=mesh.CellType.triangle,
  )

  def line(x):
    return x[0]-0.51

  V = fem.functionspace(msh, ("Lagrange", 1))

  level_set = fem.Function(V)
  level_set.interpolate(line)
  dim = msh.topology.dim

  entities_part = np.arange(11)

  intersected_entities = locate_entities_part(level_set,entities_part,dim,"phi=0")

  print(intersected_entities)

  entities_exact = np.array([2,4,7,10],dtype=np.int32)

  assert np.array_equal(intersected_entities,entities_exact)
