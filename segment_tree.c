/*
 * segment_tree.c
 *
 * C implementation of SumSegmentTree and MinSegmentTree for the Ape-X
 * prioritized replay buffer, callable from Python via ctypes.
 *
 * OpenACC annotations are included on the batch update path
 * (st_sum_set_batch / st_min_set_batch).  When compiled WITHOUT OpenACC
 * (standard gcc/clang) the pragmas are silently ignored, so the same
 * source file works in both CPU-only and GPU-accelerated builds.
 *
 * Compile (CPU only, for testing):
 *   gcc -O2 -shared -fPIC -o segment_tree.so segment_tree.c -lm
 *
 * Compile (with OpenACC via NVIDIA HPC SDK):
 *   nvc -acc -gpu=managed -O2 -shared -fPIC -o segment_tree.so segment_tree.c -lm
 *
 * Python usage: see segment_tree_ctypes.py
 */

#include <float.h>   /* DBL_MAX */
#include <math.h>    /* fmin, fmax */
#include <stdlib.h>
#include <string.h>  /* memset */

/* ── Tree struct ─────────────────────────────────────────────────────────── */

typedef struct {
    double *data;   /* heap array, length = 2 * capacity                    */
    int     cap;    /* must be a power of two                                */
} Tree;

/* ── Internal helpers ────────────────────────────────────────────────────── */

/* Propagate a changed leaf up to the root.
   op == 0  →  sum (addition)
   op == 1  →  min                                                          */
static void _propagate(Tree *t, int leaf_idx, int op) {
    int i = leaf_idx >> 1;   /* parent */
    while (i >= 1) {
        if (op == 0)
            t->data[i] = t->data[2*i] + t->data[2*i+1];
        else
            t->data[i] = fmin(t->data[2*i], t->data[2*i+1]);
        i >>= 1;
    }
}

/* ── Exported C API ──────────────────────────────────────────────────────── */

/* Allocate a tree of the requested capacity.
   Returns a heap pointer — caller must free with st_free().               */
Tree *st_alloc(int capacity) {
    Tree *t = (Tree *)malloc(sizeof(Tree));
    t->cap  = capacity;
    t->data = (double *)malloc(2 * capacity * sizeof(double));
    return t;
}

void st_free(Tree *t) {
    free(t->data);
    free(t);
}

/* Fill the tree with a uniform value (used during initialisation).        */
void st_fill(Tree *t, double val) {
    for (int i = 0; i < 2 * t->cap; i++)
        t->data[i] = val;
}

/* ── Single-element set ──────────────────────────────────────────────────── */

void st_sum_set(Tree *t, int idx, double val) {
    int leaf = idx + t->cap;
    t->data[leaf] = val;
    _propagate(t, leaf, 0);
}

void st_min_set(Tree *t, int idx, double val) {
    int leaf = idx + t->cap;
    t->data[leaf] = val;
    _propagate(t, leaf, 1);
}

/* ── Batch set (OpenACC offload path) ────────────────────────────────────── *
 *
 * Writing all leaf values first, then rebuilding the internal nodes in a
 * bottom-up sweep, is more OpenACC-friendly than calling _propagate() per
 * element, because the leaf writes are fully independent (parallel), and
 * each level of the bottom-up sweep is also independent within that level.
 *
 * Without OpenACC  → plain nested loops, still faster than N Python calls
 *                    because there is no interpreter overhead.
 * With OpenACC     → leaf writes offloaded to GPU; bottom-up rebuild also
 *                    offloaded level by level.
 *
 * idxes[i] is the logical index (0-based, as used by Python callers).
 * vals[i]  is the new priority value (already raised to alpha by Python).
 */
void st_sum_set_batch(Tree *t,
                      const int    *idxes,
                      const double *vals,
                      int           n) {
    /* Step 1: write leaves — all independent, fully parallel. */
    #pragma acc parallel loop present_or_copyin(idxes[0:n], vals[0:n]) \
                               present_or_copy(t->data[0:2*t->cap])
    for (int i = 0; i < n; i++)
        t->data[idxes[i] + t->cap] = vals[i];

    /* Step 2: bottom-up rebuild.
       The OUTER loop over levels is SEQUENTIAL — level L reads children
       written by level 2L, so levels have a strict dependency chain.
       The INNER loop over nodes within one level is fully parallel —
       nodes at the same level are independent of each other.
       Bug that was here: #pragma acc parallel loop on the outer loop
       collapsed both loops together, running all levels simultaneously
       and reading unfinished children → wrong results.                    */
    for (int level = t->cap / 2; level >= 1; level /= 2) {
        #pragma acc parallel loop present_or_copy(t->data[0:2*t->cap])
        for (int i = level; i < 2 * level; i++)
            t->data[i] = t->data[2*i] + t->data[2*i+1];
    }
}
void st_get_batch(const Tree *t, const int *idxes, double *out, int n) {
    #pragma acc parallel loop copyin(idxes[0:n]) copyout(out[0:n])
    for (int i = 0; i < n; i++)
        out[i] = t->data[idxes[i] + t->cap];
}

void st_min_set_batch(Tree *t,
                      const int    *idxes,
                      const double *vals,
                      int           n) {
    #pragma acc parallel loop present_or_copyin(idxes[0:n], vals[0:n]) \
                               present_or_copy(t->data[0:2*t->cap])
    for (int i = 0; i < n; i++)
        t->data[idxes[i] + t->cap] = vals[i];

    /* Same fix: outer level loop sequential, inner node loop parallel. */
    for (int level = t->cap / 2; level >= 1; level /= 2) {
        #pragma acc parallel loop present_or_copy(t->data[0:2*t->cap])
        for (int i = level; i < 2 * level; i++)
            t->data[i] = fmin(t->data[2*i], t->data[2*i+1]);
    }
}

/* ── Queries ─────────────────────────────────────────────────────────────── */

double st_sum_total(const Tree *t) {
    return t->data[1];   /* root always holds the total */
}

double st_min_total(const Tree *t) {
    return t->data[1];
}

double st_get(const Tree *t, int idx) {
    return t->data[idx + t->cap];
}

/* Prefix-sum index search used by _sample_proportional.
   Finds the highest index i such that sum(arr[0..i-1]) <= prefixsum.     */
int st_find_prefixsum_idx(const Tree *t, double prefixsum) {
    int idx = 1;
    while (idx < t->cap) {
        if (t->data[2*idx] > prefixsum)
            idx = 2 * idx;
        else {
            prefixsum -= t->data[2*idx];
            idx = 2 * idx + 1;
        }
    }
    return idx - t->cap;
}

/* Batch prefix-sum search — all batch_size queries are independent.       */
void st_find_prefixsum_batch(const Tree *t,
                             const double *masses,
                             int          *out_idxes,
                             int           batch_size) {
    #pragma acc parallel loop present_or_copyin(masses[0:batch_size]) \
                               present_or_copyout(out_idxes[0:batch_size]) \
                               present_or_copyin(t->data[0:2*t->cap])
    for (int b = 0; b < batch_size; b++) {
        double pfx = masses[b];
        int    idx = 1;
        while (idx < t->cap) {
            if (t->data[2*idx] > pfx)
                idx = 2 * idx;
            else {
                pfx -= t->data[2*idx];
                idx = 2 * idx + 1;
            }
        }
        out_idxes[b] = idx - t->cap;
    }
}