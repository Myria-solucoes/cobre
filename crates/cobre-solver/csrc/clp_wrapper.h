#ifndef COBRE_CLP_WRAPPER_H
#define COBRE_CLP_WRAPPER_H

/* Thin C wrapper around the CLP (COIN-OR Linear Programming) C API for use by
 * cobre-solver FFI bindings.
 *
 * All functions use the `cobre_clp_` prefix and fixed-width types (int32_t,
 * double) for FFI safety.  Each function maps 1:1 to the corresponding CLP C
 * API call with no additional logic, with two exceptions owned by this layer:
 *   1. cobre_clp_load_problem translates IEEE ±infinity bounds to the CLP
 *      infinity sentinel (±COIN_DBL_MAX) before loading the model.
 *   2. cobre_clp_load_problem fixes the objective sense to minimization.
 *
 * This header is intentionally self-contained: it does NOT include
 * <Clp_C_Interface.h> so that Rust-facing code never needs to pull in the
 * COIN-OR internal headers.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

/** Create a CLP simplex model.  Returns an opaque pointer; caller owns it.
 *  Wraps Clp_newModel(). */
void* cobre_clp_create(void);

/** Destroy a CLP simplex model and free all associated memory.
 *  Wraps Clp_deleteModel(). */
void cobre_clp_destroy(void* model);

/* =========================================================================
 * Model Loading
 * ========================================================================= */

/** Load a complete LP into the model from column-major (CSC) data.
 *  Wraps Clp_loadProblem(), then fixes the objective sense to minimize via
 *  Clp_setOptimizationDirection(model, 1).
 *
 *  Bound translation: any entry in col_lower / col_upper / row_lower /
 *  row_upper equal to IEEE +infinity is mapped to +COIN_DBL_MAX and any
 *  -infinity to -COIN_DBL_MAX before the call; finite bounds pass through
 *  unchanged.  CLP uses COIN_DBL_MAX (not IEEE infinity) as its infinity
 *  sentinel — passing raw IEEE infinity would corrupt the model.
 *
 *  num_cols, num_rows: problem dimensions.
 *  col_starts: length num_cols+1 CSC column-start offsets.
 *  row_indices, values: length col_starts[num_cols] CSC entries.
 *  col_lower, col_upper, objective: length num_cols column data.
 *  row_lower, row_upper: length num_rows row bounds. */
void cobre_clp_load_problem(
    void*           model,
    int32_t         num_cols,
    int32_t         num_rows,
    const int32_t*  col_starts,
    const int32_t*  row_indices,
    const double*   values,
    const double*   col_lower,
    const double*   col_upper,
    const double*   objective,
    const double*   row_lower,
    const double*   row_upper
);

/* =========================================================================
 * Solving
 * ========================================================================= */

/** Run the dual simplex algorithm.
 *  Wraps Clp_dual().
 *  if_values_pass is forwarded verbatim (0 = no values pass).
 *  Returns the CLP solve status int (0 = optimal); the Rust layer interprets
 *  it. */
int32_t cobre_clp_dual(void* model, int32_t if_values_pass);

/* =========================================================================
 * Solution Extraction
 * ========================================================================= */

/** Get the objective function value.
 *  Wraps Clp_objectiveValue(). */
double cobre_clp_objective_value(const void* model);

/** Get the problem status.
 *  Wraps Clp_status().
 *  0 = optimal, 1 = primal infeasible, 2 = dual infeasible,
 *  3 = stopped on iterations etc, 4 = stopped due to errors. */
int32_t cobre_clp_status(const void* model);

/** Get the simplex iteration count from the most recent solve.
 *  Wraps Clp_numberIterations(). */
int32_t cobre_clp_number_iterations(const void* model);

/** Get the primal column solution.
 *  Wraps Clp_getColSolution().
 *  Returns a pointer into CLP-owned memory (length num_cols) valid only until
 *  the next solve; do NOT free it. */
const double* cobre_clp_get_col_solution(const void* model);

/** Get the dual row solution (row prices).
 *  Wraps Clp_getRowPrice().
 *  Returns a pointer into CLP-owned memory (length num_rows) valid only until
 *  the next solve; do NOT free it. */
const double* cobre_clp_get_row_price(const void* model);

/** Get the reduced costs (dual column solution).
 *  Wraps Clp_getReducedCost().
 *  Returns a pointer into CLP-owned memory (length num_cols) valid only until
 *  the next solve; do NOT free it. */
const double* cobre_clp_get_reduced_cost(const void* model);

/* =========================================================================
 * Configuration
 * ========================================================================= */

/** Set the perturbation mode.
 *  Wraps Clp_setPerturbation().
 *  (50 = on, 100 = auto, 102 = off; see CLP docs.) */
void cobre_clp_set_perturbation(void* model, int32_t value);

/** Set the scaling mode.
 *  Wraps Clp_scaling().
 *  (0 = off, 1 = equilibrium, 2 = geometric, 3 = auto, 4 = dynamic.) */
void cobre_clp_scaling(void* model, int32_t mode);

/** Set the primal feasibility tolerance.
 *  Wraps Clp_setPrimalTolerance(). */
void cobre_clp_set_primal_tolerance(void* model, double value);

/** Set the dual feasibility tolerance.
 *  Wraps Clp_setDualTolerance(). */
void cobre_clp_set_dual_tolerance(void* model, double value);

/** Set the maximum number of simplex iterations.
 *  Wraps Clp_setMaximumIterations(). */
void cobre_clp_set_maximum_iterations(void* model, int32_t value);

/* =========================================================================
 * Basis Management
 *
 * Status values follow ClpSimplex.hpp: 0 = free, 1 = basic, 2 = at upper,
 * 3 = at lower, 4 = superbasic, 5 = fixed.
 * ========================================================================= */

/** Get the basis status of a column (variable).
 *  Wraps Clp_getColumnStatus(). */
int32_t cobre_clp_get_column_status(const void* model, int32_t sequence);

/** Set the basis status of a column (variable).
 *  Wraps Clp_setColumnStatus(). */
void cobre_clp_set_column_status(void* model, int32_t sequence, int32_t value);

/** Get the basis status of a row (slack).
 *  Wraps Clp_getRowStatus(). */
int32_t cobre_clp_get_row_status(const void* model, int32_t sequence);

/** Set the basis status of a row (slack).
 *  Wraps Clp_setRowStatus(). */
void cobre_clp_set_row_status(void* model, int32_t sequence, int32_t value);

/* =========================================================================
 * Version query (no instance required)
 * ========================================================================= */

/** Return the CLP major version number.
 *  Wraps Clp_VersionMajor(). */
int32_t cobre_clp_version_major(void);

/** Return the CLP minor version number.
 *  Wraps Clp_VersionMinor(). */
int32_t cobre_clp_version_minor(void);

/** Return the CLP release version number.
 *  Wraps Clp_VersionRelease(). */
int32_t cobre_clp_version_release(void);

#ifdef __cplusplus
}
#endif

#endif /* COBRE_CLP_WRAPPER_H */
