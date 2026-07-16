/* D3 spike: empirical run-to-run determinism probe for HiGHS 1.13.1 MIP.
 *
 * Builds seeded UC-like MIP instances (binary commitment u[g,t], continuous
 * dispatch p[g,t], startup s[g,t]; capacity/min-gen/min-up/min-down/demand
 * rows; SYMMETRIC unit pairs so multiple optimal integer solutions exist),
 * solves each R times in fresh Highs instances, and compares objective bit
 * patterns and an FNV-1a hash of the full solution vector.
 *
 * Modes: threads=1 (the doc's proposed policy) and threads=8 + parallel=on.
 * Also builds a column-PERMUTED copy of instance 0 to document order
 * sensitivity (expected: possibly different optimum vertex -> supports the
 * canonical-ordering requirement, not a failure).
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "interfaces/highs_c_api.h"

/* deterministic LCG so instances are identical across runs/processes */
static uint64_t lcg_state;
static void lcg_seed(uint64_t s) { lcg_state = s * 2862933555777941757ULL + 3037000493ULL; }
static double lcg01(void) {
  lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)(lcg_state >> 11) / 9007199254740992.0;
}

static uint64_t fnv1a(const void* data, size_t n) {
  const unsigned char* p = (const unsigned char*)data;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}

typedef struct {
  HighsInt ncol, nrow, nnz;
  double *cost, *cl, *cu, *rl, *ru, *aval;
  HighsInt *astart, *aindex;  /* rowwise */
  HighsInt *integrality;
} Mip;

/* row buffer */
static double rbuf_v[4096];
static HighsInt rbuf_i[4096];
static int rbuf_n;
static void row_reset(void) { rbuf_n = 0; }
static void row_add(HighsInt col, double v) { rbuf_i[rbuf_n] = col; rbuf_v[rbuf_n] = v; rbuf_n++; }

static void mip_row(Mip* m, double lo, double hi) {
  for (int k = 0; k < rbuf_n; k++) {
    m->aindex[m->nnz] = rbuf_i[k];
    m->aval[m->nnz] = rbuf_v[k];
    m->nnz++;
  }
  m->rl[m->nrow] = lo; m->ru[m->nrow] = hi;
  m->nrow++;
  m->astart[m->nrow] = m->nnz;
}

/* UC instance: G units, T periods, min-up/down L. perm = optional column
 * permutation (identity if NULL). */
static Mip build_uc(int G, int T, int L, uint64_t seed, const int* perm) {
  int ncol = 3 * G * T; /* u, p, s per (g,t) */
  int max_rows = G * T * (3 + 2 * (L - 1) + 1) + 2 * T + 8;
  int max_nnz = max_rows * 8;
  Mip m = {0};
  m.ncol = ncol;
  m.cost = calloc(ncol, sizeof(double));
  m.cl = calloc(ncol, sizeof(double));
  m.cu = calloc(ncol, sizeof(double));
  m.rl = calloc(max_rows, sizeof(double));
  m.ru = calloc(max_rows, sizeof(double));
  m.aval = calloc(max_nnz, sizeof(double));
  m.astart = calloc(max_rows + 1, sizeof(HighsInt));
  m.aindex = calloc(max_nnz, sizeof(HighsInt));
  m.integrality = calloc(ncol, sizeof(HighsInt));

#define U(g, t) (perm ? perm[3 * ((g) * T + (t)) + 0] : 3 * ((g) * T + (t)) + 0)
#define P(g, t) (perm ? perm[3 * ((g) * T + (t)) + 1] : 3 * ((g) * T + (t)) + 1)
#define S(g, t) (perm ? perm[3 * ((g) * T + (t)) + 2] : 3 * ((g) * T + (t)) + 2)

  lcg_seed(seed);
  double* pmax = calloc(G, sizeof(double));
  double* pmin = calloc(G, sizeof(double));
  double* cvar = calloc(G, sizeof(double));
  double* cstart = calloc(G, sizeof(double));
  double* cnoload = calloc(G, sizeof(double));
  double cap_tot = 0.0;
  for (int g = 0; g < G; g++) {
    if (g % 4 != 0) { /* symmetric group of 4 -> many optima */
      int b0 = g - (g % 4);
      pmax[g] = pmax[b0]; pmin[g] = pmin[b0]; cvar[g] = cvar[b0];
      cstart[g] = cstart[b0]; cnoload[g] = cnoload[b0];
    } else {
      pmax[g] = 60.0 + 240.0 * lcg01();
      pmin[g] = 0.45 * pmax[g];
      cvar[g] = 20.0 + 6.0 * lcg01();
      cstart[g] = 200.0 + 1800.0 * lcg01();
      cnoload[g] = 30.0 + 120.0 * lcg01();
    }
    cap_tot += pmax[g];
  }
  for (int g = 0; g < G; g++)
    for (int t = 0; t < T; t++) {
      m.integrality[U(g, t)] = kHighsVarTypeInteger;
      m.cl[U(g, t)] = 0.0; m.cu[U(g, t)] = 1.0; m.cost[U(g, t)] = cnoload[g];
      m.cl[P(g, t)] = 0.0; m.cu[P(g, t)] = pmax[g]; m.cost[P(g, t)] = cvar[g];
      m.cl[S(g, t)] = 0.0; m.cu[S(g, t)] = 1.0; m.cost[S(g, t)] = cstart[g];
    }

  m.astart[0] = 0;
  /* demand rows */
  for (int t = 0; t < T; t++) {
    double shape = 0.55 + 0.4 * lcg01();
    double demand = shape * 0.72 * cap_tot;
    row_reset();
    for (int g = 0; g < G; g++) row_add(P(g, t), 1.0);
    mip_row(&m, demand, 1e30);
    row_reset();
    for (int g = 0; g < G; g++) row_add(U(g, t), pmax[g]);
    mip_row(&m, 1.06 * demand, 1e30);
  }
  for (int g = 0; g < G; g++)
    for (int t = 0; t < T; t++) {
      /* p <= pmax * u ; p >= pmin * u */
      row_reset(); row_add(P(g, t), 1.0); row_add(U(g, t), -pmax[g]);
      mip_row(&m, -1e30, 0.0);
      row_reset(); row_add(P(g, t), 1.0); row_add(U(g, t), -pmin[g]);
      mip_row(&m, 0.0, 1e30);
      if (t > 0) {
        /* ramp-up: p_t - p_{t-1} - pmax*s <= 0.25*pmax */
        row_reset(); row_add(P(g, t), 1.0); row_add(P(g, t - 1), -1.0); row_add(S(g, t), -pmax[g]);
        mip_row(&m, -1e30, 0.25 * pmax[g]);
        /* startup: s >= u_t - u_{t-1} */
        row_reset(); row_add(S(g, t), 1.0); row_add(U(g, t), -1.0); row_add(U(g, t - 1), 1.0);
        mip_row(&m, 0.0, 1e30);
        for (int tau = t + 1; tau < t + L && tau < T; tau++) {
          /* min-up: u_t - u_{t-1} <= u_tau */
          row_reset(); row_add(U(g, t), 1.0); row_add(U(g, t - 1), -1.0); row_add(U(g, tau), -1.0);
          mip_row(&m, -1e30, 0.0);
          /* min-down: u_{t-1} - u_t + u_tau <= 1 */
          row_reset(); row_add(U(g, t - 1), 1.0); row_add(U(g, t), -1.0); row_add(U(g, tau), 1.0);
          mip_row(&m, -1e30, 1.0);
        }
      }
    }
  free(pmax); free(pmin); free(cvar); free(cstart); free(cnoload);
  return m;
}

static void mip_free(Mip* m) {
  free(m->cost); free(m->cl); free(m->cu); free(m->rl); free(m->ru);
  free(m->aval); free(m->astart); free(m->aindex); free(m->integrality);
}

typedef struct {
  uint64_t obj_bits, sol_hash;
  int64_t nodes;
  int status;
} RunResult;

static int g_stress = 0;
static int g_nodecap = 3000;
static RunResult solve_once(const Mip* m, int threads) {
  void* h = Highs_create();
  Highs_setBoolOptionValue(h, "output_flag", 0);
  Highs_setIntOptionValue(h, "threads", threads);
  Highs_setDoubleOptionValue(h, "mip_rel_gap", 0.0);
  Highs_setDoubleOptionValue(h, "mip_abs_gap", 0.0);
  Highs_setIntOptionValue(h, "mip_max_nodes", g_nodecap);
  if (threads > 1) Highs_setStringOptionValue(h, "parallel", "on");
  if (g_stress & 1) Highs_setStringOptionValue(h, "presolve", "off");
  if (g_stress & 2) Highs_setBoolOptionValue(h, "mip_detect_symmetry", 0);
  RunResult r = {0};
  Highs_passMip(h, m->ncol, m->nrow, m->nnz, kHighsMatrixFormatRowwise,
                kHighsObjSenseMinimize, 0.0, m->cost, m->cl, m->cu, m->rl,
                m->ru, m->astart, m->aindex, m->aval, m->integrality);
  int status = Highs_run(h);
  double obj = Highs_getObjectiveValue(h);
  double* sol = calloc(m->ncol, sizeof(double));
  Highs_getSolution(h, sol, NULL, NULL, NULL);
  memcpy(&r.obj_bits, &obj, 8);
  r.sol_hash = fnv1a(sol, m->ncol * sizeof(double));
  Highs_getInt64InfoValue(h, "mip_node_count", &r.nodes);
  r.status = status;
  free(sol);
  Highs_destroy(h);
  return r;
}

int main(int argc, char** argv) {
  int R = argc > 1 ? atoi(argv[1]) : 8;
  int threads_mode = argc > 2 ? atoi(argv[2]) : 1;
  g_stress = argc > 3 ? atoi(argv[3]) : 0;
  g_nodecap = argc > 4 ? atoi(argv[4]) : 3000;
  int only_case = argc > 5 ? atoi(argv[5]) : -1;
  struct { int G, T; uint64_t seed; } cases[] = {
    {24, 48, 101}, {32, 48, 303}, {28, 60, 505},
  };
  int ncases = 3;
  int all_ok = 1;
  for (int c = 0; c < ncases; c++) {
    if (only_case >= 0 && c != only_case) continue;
    Mip m = build_uc(cases[c].G, cases[c].T, 3, cases[c].seed, NULL);
    RunResult first = {0};
    int ok = 1;
    for (int r = 0; r < R; r++) {
      RunResult rr = solve_once(&m, threads_mode);
      if (r == 0) first = rr;
      else if (rr.obj_bits != first.obj_bits || rr.sol_hash != first.sol_hash ||
               rr.nodes != first.nodes)
        ok = 0;
      printf("case=%d G=%d T=%d run=%d threads=%d status=%d obj_bits=%016llx sol_hash=%016llx nodes=%lld\n",
             c, cases[c].G, cases[c].T, r, threads_mode, rr.status,
             (unsigned long long)rr.obj_bits, (unsigned long long)rr.sol_hash,
             (long long)rr.nodes);
      fflush(stdout);
    }
    printf("case=%d VERDICT threads=%d: %s\n", c, threads_mode,
           ok ? "BIT-IDENTICAL across runs" : "NONDETERMINISTIC");
    if (!ok) all_ok = 0;
    mip_free(&m);
  }

  /* order-sensitivity probe: permuted columns of case 0 (fresh instance) */
  if (threads_mode == 1) {
    int G = 10, T = 24;
    int ncol = 3 * G * T;
    int* perm = calloc(ncol, sizeof(int));
    for (int i = 0; i < ncol; i++) perm[i] = i;
    lcg_seed(999);
    for (int i = ncol - 1; i > 0; i--) {
      int j = (int)(lcg01() * (i + 1));
      int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    Mip mp = build_uc(G, T, 3, 101, perm);
    RunResult rp = solve_once(&mp, 1);
    Mip m0 = build_uc(G, T, 3, 101, NULL);
    RunResult r0 = solve_once(&m0, 1);
    double op, oo;
    memcpy(&op, &rp.obj_bits, 8); memcpy(&oo, &r0.obj_bits, 8);
    printf("permutation probe: obj_canonical=%.17g obj_permuted=%.17g objbits_equal=%d nodes %lld vs %lld\n",
           oo, op, rp.obj_bits == r0.obj_bits, (long long)r0.nodes, (long long)rp.nodes);
    mip_free(&mp); mip_free(&m0);
    free(perm);
  }
  printf("OVERALL threads=%d: %s\n", threads_mode,
         all_ok ? "DETERMINISTIC (run-to-run, fixed order)" : "NONDETERMINISM OBSERVED");
  return all_ok ? 0 : 1;
}
