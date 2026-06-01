# DOF Constraints

CutFEMx solves PDEs using finite element spaces defined on a background mesh.
Only part of that mesh may intersect the physical domain. Degrees of freedom
whose basis functions do not touch the active domain must be constrained before
solving; otherwise the assembled matrix can contain zero rows or uncontrolled
nullspace components.

```{admonition} Active and inactive degrees of freedom
:class: theory

Let $V_h=\operatorname{span}\{\varphi_i\}$ be the finite element space on the
background mesh. For a problem posed on $\Omega_h$, the physically active index
set is

$$
{\mathcal I}_\Omega
=
\{i:\operatorname{supp}(\varphi_i)\cap\Omega_h\ne\emptyset\}.
$$

An index $i\notin{\mathcal I}_\Omega$ has no physical volume, boundary, or
stabilization contribution. The corresponding row in the assembled system is
therefore not determined by the PDE unless it is constrained algebraically.
```

## Active Domain From A Form

After compiling a CutFEMx form, derive the active domain from that form:

```python
a_form = cutfemx.fem.form(a)
L_form = cutfemx.fem.form(L)

active_domain = cutfemx.fem.active_domain(a_form)
```

The active-domain object is derived from the integration domains of the form.
This is important: the active degrees of freedom are not determined by the
level set alone, but by the measures that actually appear in the assembled
problem.

```{admonition} Form-dependent activity
:class: theory

If a form contains integrals only over $\Omega_h$, then activity is determined
by basis functions with support on $\Omega_h$. If additional skeleton or
stabilization terms are present, the active support may include neighbouring
cells touched by those terms. The active-domain query therefore belongs after
form construction, not before it.
```

## Deactivation

For MatrixCSR assembly, assemble and deactivate outside the active domain:

```python
b = cutfemx.fem.assemble_vector(L_form)
A = cutfemx.fem.assemble_matrix(a_form)

cutfemx.fem.deactivate_outside(A, b, active_domain)
```

For PETSc assembly, use the PETSc helpers:

```python
import cutfemx.petsc

A = cutfemx.petsc.assemble_matrix(a_form)
b = cutfemx.petsc.assemble_vector(L_form)

cutfemx.petsc.deactivate_outside(A, b, active_domain)
```

```{admonition} What deactivation does
:class: practice

For an inactive degree of freedom $i$, deactivation replaces the row by a
simple identity constraint,

$$
A_{ii}=1,\qquad
A_{ij}=0\quad(j\ne i),\qquad
b_i=0.
$$

This keeps the linear system nonsingular while leaving the active finite
element problem unchanged. Active rows are not modified by this operation.
```

## Block Systems

Interface and multi-domain problems often use one finite element space per
phase. The active set is then block dependent. For a two-phase problem with
unknowns $(u_1,u_2)$, the algebraic system has the form

$$
\begin{bmatrix}
A_{11} & A_{12}\\
A_{21} & A_{22}
\end{bmatrix}
\begin{bmatrix}
U_1\\
U_2
\end{bmatrix}
=
\begin{bmatrix}
B_1\\
B_2
\end{bmatrix}.
$$

The inactive degrees of freedom for $u_1$ and $u_2$ are generally different,
because $\Omega_1$ and $\Omega_2$ occupy different parts of the background
mesh. Use block deactivation for such systems:

```python
domain1 = cutfemx.fem.active_domain(a_forms[0][0])
domain2 = cutfemx.fem.active_domain(a_forms[1][1])

cutfemx.fem.deactivate_outside_blocks(
    A_matrix_blocks,
    [domain1, domain2],
    b_vector_blocks,
)
```

The interface Poisson example uses this pattern.

## Diagnostics

The active-domain object exposes arrays that are useful for debugging:

```python
active_cells = active_domain.active_cells
inactive_dofs = active_domain.inactive_dofs
indicator = active_domain.indicator
```

`indicator` is a DOLFINx function that can be visualized on the background
mesh. It is often the fastest way to detect a selector error: if the active
region is empty, shifted, or inverted, the inactive-DOF pattern will show it
immediately.

After deactivation, zero-row diagnostics can be used as a final algebraic
check:

```python
zero_rows = cutfemx.fem.zero_rows(A)
```

For a well-constrained scalar problem, no zero rows should remain after
deactivation. In block systems, use the corresponding block-row diagnostic.
