# -----------------------------------------------------------------------------
# Copyright (c) 2022 Nana Ofori-Opoku. All rights reserved.

import time
import dolfin as df
import amr_fenics.helpers as hlp
import amr_fenics.input_output as io

# Fenics form parameters that control adaption  behaviour
# df.parameters["mesh_partitioner"] = "SCOTCH"

df.parameters["mesh_partitioner"] = "ParMETIS"
df.parameters["partitioning_approach"] = "ADAPTIVE_REPARTITION"

# Alternatives for ParMETIS
# df.parameters["partitioning_approach"] = "PARTITION"
# df.parameters["partitioning_approach"] = "REFINE"

# Choose refinement algorithm (needed for solver refinement algorithm)
df.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
# df.parameters['ghost_mode'] = 'shared_vertex'
# df.parameters['ghost_mode'] = 'shared_facet'
#df.parameters["ghost_mode"] = "None"


def calc_resolution(maximum_element, smallest_element):
    resolution = int(df.ln(2 * maximum_element / smallest_element) / df.ln(2.0))
    io.disp(f"├─ Number of levels of mesh resolution is {resolution}")
    return resolution


def initial_mesh(mesh, resolution):
    for res in range(resolution - 1):
        mesh = df.refine(mesh, True)

    cells = df.MPI.sum(hlp.mpiComm, mesh.num_cells())
    hmin = df.MPI.min(hlp.mpiComm, mesh.hmin())
    hmax = df.MPI.max(hlp.mpiComm, mesh.hmax())

    io.disp(f"├─ Initial mesh: {cells} cells with min(𝒉)={hmin:<g}, max(𝒉)={hmax:<g}")
    return mesh


def initial_mesh_elect(mesh, resolution, thickness):
    for res in range(resolution - 1):
        cell_markers_refine = df.MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers_refine.set_all(False)

        for cell in df.cells(mesh):
            y = cell.midpoint().y()
            if y < thickness:
                cell_markers_refine[cell] = True

        mesh = df.refine(mesh, cell_markers_refine, True)

    cells = df.MPI.sum(hlp.mpiComm, mesh.num_cells())
    hmin = df.MPI.min(hlp.mpiComm, mesh.hmin())
    hmax = df.MPI.max(hlp.mpiComm, mesh.hmax())

    io.disp(
        f"├─ Initial mesh: {cells:.0f} cells with min(𝒉)={hmin:<g}, max(𝒉)={hmax:<g}"
    )
    return mesh


def refine_coarsen(
    coarse_mesh,
    current_mesh,
    element,
    resolution,
    field_vals,
    adapt_thresh,
    beta,
):
    # val_min = 5e-3
    # val_max = 9.9e-1
    val_min = 1e-1
    val_max = 9e-1
    refineStart = time.time()

    hmin = df.MPI.min(hlp.mpiComm, coarse_mesh.hmin())

    smooth_grd = calc_adapt_criteria_phaseField(current_mesh, field_vals, element, beta)
    mesh = df.Mesh(coarse_mesh)

    for res in range(resolution - 1):
        cell_markers_refine = df.MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers_refine.set_all(False)

        V = df.FunctionSpace(mesh, element)
        current_grd = df.Function(V)
        df.LagrangeInterpolator.interpolate(current_grd, smooth_grd)
        current_grd.set_allow_extrapolation(True)

        current_val = df.Function(V)
        df.LagrangeInterpolator.interpolate(current_val, field_vals.sub(0))
        current_val.set_allow_extrapolation(True)

        # Cycle through all cells of current mesh and flag cells for refinement
        for cell in df.cells(mesh):
            cv = cell.volume()
            cp = cell.midpoint()
            sum_facet = 0
            sum_val = 0
            c = 0

            for facet in df.facets(cell):
                fp = facet.midpoint()
                sum_facet += current_grd(fp)
                sum_val += current_val(fp)
                c += 1
            
            if cell.h()/2 > hmin:
                if (
                    sum_facet * cv > adapt_thresh
                    or current_grd(cp) * cv > adapt_thresh
                    or (current_val(cp) > val_min and current_val(cp) < val_max)
                    or (sum_val / c > val_min and sum_val / c < val_max)
                ):
                    cell_markers_refine[cell] = True

        mesh = df.refine(mesh, cell_markers_refine, True)

    timer = time.time() - refineStart

    cells = df.MPI.sum(hlp.mpiComm, mesh.num_cells())
    hmin = df.MPI.min(hlp.mpiComm, mesh.hmin())
    hmax = df.MPI.max(hlp.mpiComm, mesh.hmax())

    io.disp(
        f"│ ├─      Refined mesh: {cells} cells "
        f"in {timer:,.0f} s\n"
        f"│ ├─      [min(𝒉)={hmin:g}, max(𝒉)={hmax:g}]"
    )

    return mesh


def calc_adapt_criteria_phaseField(mesh, U, element, beta):
    # access to values (by reference)
    fields = U.split(deepcopy=True)

    psi = sum(
        beta[r] * df.inner(df.grad(fields[r]), df.grad(fields[r]))
        for r in range(len(beta))
    )

    grd = df.sqrt(psi)

    smooth_grd = df.project(
        grd,
        V=df.FunctionSpace(mesh, element),
        bcs=[],
        mesh=mesh,
        solver_type="lu",
        form_compiler_parameters={"quadrature_degree": 4},
    )

    return smooth_grd
