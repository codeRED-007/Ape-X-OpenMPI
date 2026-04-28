"""
test_segment_tree.py

Correctness checks and timing benchmark comparing the original pure-Python
SumSegmentTree / MinSegmentTree against the C-backed ctypes versions.

Run:
    python test_segment_tree.py
"""

import random
import sys
import time

import numpy as np

# ── Import both implementations ───────────────────────────────────────────────
# Python originals (memory.py must be on the path or in the same directory)
sys.path.insert(0, "/mnt/user-data/uploads")
from memory import SumSegmentTree as PySumTree, MinSegmentTree as PyMinTree

# C-backed replacements
sys.path.insert(0, "/home/claude")
from segment_tree_ctypes import SumSegmentTree as CSumTree, MinSegmentTree as CMinTree

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def check(label, got, expected, tol=1e-9):
    ok = abs(got - expected) <= tol
    print(f"  {PASS if ok else FAIL}  {label}: got={got:.8f}  expected={expected:.8f}")
    return ok

# ═════════════════════════════════════════════════════════════════════════════
# 1. Correctness
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. CORRECTNESS — SumSegmentTree")
print("=" * 60)

CAP = 16
py_s = PySumTree(CAP)
c_s  = CSumTree(CAP)

# Set the same random values in both trees
random.seed(42)
vals = [random.uniform(0.1, 5.0) for _ in range(CAP)]
for i, v in enumerate(vals):
    py_s[i] = v
    c_s[i]  = v

all_ok = True
all_ok &= check("total sum",   c_s.sum(),      py_s.sum())
all_ok &= check("partial 0:8", c_s.sum(0, 8),  py_s.sum(0, 8))
all_ok &= check("partial 4:12",c_s.sum(4, 12), py_s.sum(4, 12))
all_ok &= check("single get[7]",c_s[7],        py_s[7])

# prefixsum search
ps = py_s.sum() * 0.42
all_ok &= check("find_prefixsum",
                float(c_s.find_prefixsum_idx(ps)),
                float(py_s.find_prefixsum_idx(ps)))

# batch set
new_vals  = np.random.default_rng(7).uniform(0.1, 3.0, 6)
new_idxes = [1, 3, 5, 7, 9, 11]
for i, v in zip(new_idxes, new_vals):
    py_s[i] = v
c_s.set_batch(new_idxes, new_vals)
all_ok &= check("sum after batch set", c_s.sum(), py_s.sum())

print()
print("=" * 60)
print("2. CORRECTNESS — MinSegmentTree")
print("=" * 60)

py_m = PyMinTree(CAP)
c_m  = CMinTree(CAP)

random.seed(99)
vals = [random.uniform(0.1, 5.0) for _ in range(CAP)]
for i, v in enumerate(vals):
    py_m[i] = v
    c_m[i]  = v

all_ok &= check("global min",    c_m.min(),      py_m.min())
all_ok &= check("partial 0:8",   c_m.min(0, 8),  py_m.min(0, 8))
all_ok &= check("partial 4:12",  c_m.min(4, 12), py_m.min(4, 12))

new_vals  = np.random.default_rng(13).uniform(0.05, 2.0, 6)
for i, v in zip(new_idxes, new_vals):
    py_m[i] = v
c_m.set_batch(new_idxes, new_vals)
all_ok &= check("min after batch set", c_m.min(), py_m.min())

# ═════════════════════════════════════════════════════════════════════════════
# 3. Batch find_prefixsum correctness
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("3. CORRECTNESS — batch find_prefixsum_batch")
print("=" * 60)

CAP2  = 1024
py_s2 = PySumTree(CAP2)
c_s2  = CSumTree(CAP2)
rng   = np.random.default_rng(0)
vs    = rng.uniform(0.1, 10.0, CAP2)
for i, v in enumerate(vs):
    py_s2[i] = v
    c_s2[i]  = v

batch = 512
total = c_s2.sum()
masses = rng.uniform(0, total, batch)

py_idxes = np.array([py_s2.find_prefixsum_idx(m) for m in masses], dtype=np.int32)
c_idxes  = c_s2.find_prefixsum_batch(masses)
match = np.all(py_idxes == c_idxes)
print(f"  {PASS if match else FAIL}  All {batch} prefixsum indices match: {match}")
all_ok &= match

# ═════════════════════════════════════════════════════════════════════════════
# 4. Timing benchmark
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("4. BENCHMARK")
print("=" * 60)

BENCH_CAP   = 2**21   # 2 M  (matches args.replay_buffer_size)
BENCH_N     = 50      # transitions per actor send_interval
BENCH_REPS  = 1000    # repetitions

rng       = np.random.default_rng(42)
bench_idx = rng.integers(0, BENCH_CAP, BENCH_N).astype(np.int32)
bench_val = rng.uniform(0.1, 5.0, BENCH_N)

# Initialise both trees with the same data so the test is fair
py_bench = PySumTree(BENCH_CAP)
c_bench  = CSumTree(BENCH_CAP)
init_v   = rng.uniform(0.1, 5.0, BENCH_CAP)
for i, v in enumerate(init_v):
    py_bench[i] = float(v)
    c_bench[i]  = float(v)

# ── Python: single-element __setitem__ in a loop (current code path) ─────────
t0 = time.perf_counter()
for _ in range(BENCH_REPS):
    for i, v in zip(bench_idx, bench_val):
        py_bench[i] = v
py_time = time.perf_counter() - t0

# ── C: set_batch (single C call per rep) ─────────────────────────────────────
t0 = time.perf_counter()
for _ in range(BENCH_REPS):
    c_bench.set_batch(bench_idx, bench_val)
c_time = time.perf_counter() - t0

print(f"  Tree capacity : {BENCH_CAP:,}")
print(f"  Batch size    : {BENCH_N} leaves per update")
print(f"  Repetitions   : {BENCH_REPS:,}")
print()
print(f"  Python __setitem__ loop : {py_time*1000:.1f} ms total  ({py_time/BENCH_REPS*1e6:.1f} µs/batch)")
print(f"  C set_batch             : {c_time*1000:.1f} ms total  ({c_time/BENCH_REPS*1e6:.1f} µs/batch)")
print(f"  Speedup                 : {py_time/c_time:.1f}×")

# ── Batch prefixsum ────────────────────────────────────────────────────────────
SAMPLE_BATCH = 512
total        = c_bench.sum()
masses_bench = rng.uniform(0, total, SAMPLE_BATCH)

t0 = time.perf_counter()
for _ in range(BENCH_REPS):
    [py_bench.find_prefixsum_idx(m) for m in masses_bench]
py_pfx_time = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(BENCH_REPS):
    c_bench.find_prefixsum_batch(masses_bench)
c_pfx_time = time.perf_counter() - t0

print()
print(f"  Prefixsum batch size    : {SAMPLE_BATCH}")
print(f"  Python loop             : {py_pfx_time*1000:.1f} ms  ({py_pfx_time/BENCH_REPS*1e6:.1f} µs/batch)")
print(f"  C find_prefixsum_batch  : {c_pfx_time*1000:.1f} ms  ({c_pfx_time/BENCH_REPS*1e6:.1f} µs/batch)")
print(f"  Speedup                 : {py_pfx_time/c_pfx_time:.1f}×")

print()
print("=" * 60)
print(f"  Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
print("=" * 60)
