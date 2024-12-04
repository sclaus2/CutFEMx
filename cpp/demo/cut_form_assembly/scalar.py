# The first step is to define the variational problem at hand. We define
# the variational problem in UFL terms in a separate form file
# {download}`demo_poisson/poisson.py`.  We begin by defining the finite
# element:

from basix.ufl import element
from ufl import (
    Constant,
    FunctionSpace,
    Mesh,
    dx,
)

e = element("Lagrange", "triangle", 1)

# The first argument to :py:class:`FiniteElement` is the finite element
# family, the second argument specifies the domain, while the third
# argument specifies the polynomial degree. Thus, in this case, our
# element `element` consists of first-order, continuous Lagrange basis
# functions on triangles (or in order words, continuous piecewise linear
# polynomials on triangles).
#
# Next, we use this element to initialize the trial and test functions
# ($u$ and $v$) and the coefficient functions ($f$ and $g$):

coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, e)
alpha = Constant(mesh)


# Finally, we define the bilinear and linear forms according to the
# variational formulation of the equations:
L = alpha*(dx(0, scheme="runtime")+dx(0))
