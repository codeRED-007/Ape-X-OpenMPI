"""
segment_tree_ctypes.py

Drop-in replacements for SumSegmentTree and MinSegmentTree from memory.py,
backed by the C shared library (segment_tree.so) via ctypes.

The public API is identical to the Python originals so CustomPrioritizedReplayBuffer
works without any changes — just swap the import.

Key additions over the pure-Python version:
  - set_batch(idxes, vals) : update many leaves in one C call
                             (uses OpenACC parallel loops when compiled with nvc)
  - find_prefixsum_batch(masses) : sample all batch_size indices in one C call

How to use
----------
    # In memory.py, replace:
    #   from memory import SumSegmentTree, MinSegmentTree
    # with:
    from segment_tree_ctypes import SumSegmentTree, MinSegmentTree

Compile the .so first:
    gcc -O2 -shared -fPIC -o segment_tree.so segment_tree.c -lm
    # or with OpenACC:
    nvc -acc -gpu=managed -O2 -shared -fPIC -o segment_tree.so segment_tree.c -lm
"""

import ctypes
import os
import pathlib

import numpy as np

# ── Load the shared library ───────────────────────────────────────────────────
_LIB_PATH = pathlib.Path(__file__).parent / "segment_tree.so"
_lib = ctypes.CDLL(str(_LIB_PATH))

# ── ctypes struct mirroring the C Tree struct ─────────────────────────────────
class _CTree(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("cap",  ctypes.c_int),
    ]

# ── Function signatures ───────────────────────────────────────────────────────
_lib.st_alloc.restype  = ctypes.POINTER(_CTree)
_lib.st_alloc.argtypes = [ctypes.c_int]

_lib.st_free.restype  = None
_lib.st_free.argtypes = [ctypes.POINTER(_CTree)]

_lib.st_fill.restype  = None
_lib.st_fill.argtypes = [ctypes.POINTER(_CTree), ctypes.c_double]

_lib.st_sum_set.restype  = None
_lib.st_sum_set.argtypes = [ctypes.POINTER(_CTree), ctypes.c_int, ctypes.c_double]

_lib.st_min_set.restype  = None
_lib.st_min_set.argtypes = [ctypes.POINTER(_CTree), ctypes.c_int, ctypes.c_double]

_lib.st_sum_set_batch.restype  = None
_lib.st_sum_set_batch.argtypes = [
    ctypes.POINTER(_CTree),
    ctypes.POINTER(ctypes.c_int),     # idxes
    ctypes.POINTER(ctypes.c_double),  # vals
    ctypes.c_int,                     # n
]

_lib.st_min_set_batch.restype  = None
_lib.st_min_set_batch.argtypes = [
    ctypes.POINTER(_CTree),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]

_lib.st_sum_total.restype  = ctypes.c_double
_lib.st_sum_total.argtypes = [ctypes.POINTER(_CTree)]

_lib.st_min_total.restype  = ctypes.c_double
_lib.st_min_total.argtypes = [ctypes.POINTER(_CTree)]

_lib.st_get.restype  = ctypes.c_double
_lib.st_get.argtypes = [ctypes.POINTER(_CTree), ctypes.c_int]

_lib.st_find_prefixsum_idx.restype  = ctypes.c_int
_lib.st_find_prefixsum_idx.argtypes = [ctypes.POINTER(_CTree), ctypes.c_double]

_lib.st_find_prefixsum_batch.restype  = None
_lib.st_find_prefixsum_batch.argtypes = [
    ctypes.POINTER(_CTree),
    ctypes.POINTER(ctypes.c_double),  # masses
    ctypes.POINTER(ctypes.c_int),     # out_idxes
    ctypes.c_int,                     # batch_size
]


# ── Helper: numpy array → ctypes pointer (zero-copy) ─────────────────────────
def _int_ptr(arr: np.ndarray):
    a = np.ascontiguousarray(arr, dtype=np.int32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), a

def _dbl_ptr(arr: np.ndarray):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), a


# ── SumSegmentTree ────────────────────────────────────────────────────────────
class SumSegmentTree:
    """
    Drop-in replacement for memory.SumSegmentTree backed by C + optional OpenACC.

    Public API matches the original exactly.
    Extra methods: set_batch(), find_prefixsum_batch().
    """

    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, \
            "capacity must be a power of 2"
        self._capacity = capacity
        self._tree = _lib.st_alloc(capacity)
        _lib.st_fill(self._tree, 0.0)   # neutral element for sum

    def __del__(self):
        if hasattr(self, "_tree") and self._tree:
            _lib.st_free(self._tree)

    # ── Single-element interface (matches original __setitem__ / __getitem__) --

    def __setitem__(self, idx: int, val: float):
        assert 0 <= idx < self._capacity
        _lib.st_sum_set(self._tree, idx, float(val))

    def __getitem__(self, idx: int) -> float:
        assert 0 <= idx < self._capacity
        return _lib.st_get(self._tree, idx)

    # ── Batch interface (new — replaces a Python for-loop in update_priorities) -

    def set_batch(self, idxes, vals):
        """
        Set tree[idxes[i]] = vals[i] for all i in one C call.
        idxes: array-like of int   (logical 0-based indices)
        vals:  array-like of float (already raised to alpha by the caller)
        """
        idxes_np = np.asarray(idxes, dtype=np.int32)
        vals_np  = np.asarray(vals,  dtype=np.float64)
        ip, _ia  = _int_ptr(idxes_np)
        dp, _da  = _dbl_ptr(vals_np)
        _lib.st_sum_set_batch(self._tree, ip, dp, len(idxes_np))

    # ── Queries (match original SumSegmentTree) ───────────────────────────────

    def sum(self, start: int = 0, end=None) -> float:
        """sum(arr[start] + ... + arr[end-1])"""
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        # For full-range queries (the common case) use the O(1) root read.
        if start == 0 and end == self._capacity:
            return _lib.st_sum_total(self._tree)
        # Partial range: fall back to the recursive C helper via Python.
        # (Rare in practice — _sample_proportional always uses full range.)
        return self._reduce(start, end - 1, 1, 0, self._capacity - 1)

    def _reduce(self, start, end, node, node_start, node_end):
        """Mirror of Python _reduce_helper, reading from the C tree data."""
        data = self._tree.contents.data
        if start == node_start and end == node_end:
            return data[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce(start, end, 2*node, node_start, mid)
        elif mid + 1 <= start:
            return self._reduce(start, end, 2*node+1, mid+1, node_end)
        else:
            return (self._reduce(start, mid, 2*node, node_start, mid) +
                    self._reduce(mid+1, end, 2*node+1, mid+1, node_end))

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Single prefix-sum search (used by _sample_proportional)."""
        return _lib.st_find_prefixsum_idx(self._tree, prefixsum)

    def find_prefixsum_batch(self, masses) -> np.ndarray:
        """
        Batch prefix-sum search — all batch_size queries in one C call.
        masses: array-like of float, shape (batch_size,)
        Returns: np.ndarray of int32, shape (batch_size,)
        """
        masses_np = np.asarray(masses, dtype=np.float64)
        out       = np.empty(len(masses_np), dtype=np.int32)
        mp, _ma   = _dbl_ptr(masses_np)
        op, _oa   = _int_ptr(out)
        _lib.st_find_prefixsum_batch(self._tree, mp, op, len(masses_np))
        return out


# ── MinSegmentTree ────────────────────────────────────────────────────────────
class MinSegmentTree:
    """
    Drop-in replacement for memory.MinSegmentTree backed by C + optional OpenACC.
    """

    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, \
            "capacity must be a power of 2"
        self._capacity = capacity
        self._tree = _lib.st_alloc(capacity)
        _lib.st_fill(self._tree, float("inf"))  # neutral element for min

    def __del__(self):
        if hasattr(self, "_tree") and self._tree:
            _lib.st_free(self._tree)

    def __setitem__(self, idx: int, val: float):
        assert 0 <= idx < self._capacity
        _lib.st_min_set(self._tree, idx, float(val))

    def __getitem__(self, idx: int) -> float:
        assert 0 <= idx < self._capacity
        return _lib.st_get(self._tree, idx)

    def set_batch(self, idxes, vals):
        idxes_np = np.asarray(idxes, dtype=np.int32)
        vals_np  = np.asarray(vals,  dtype=np.float64)
        ip, _ia  = _int_ptr(idxes_np)
        dp, _da  = _dbl_ptr(vals_np)
        _lib.st_min_set_batch(self._tree, ip, dp, len(idxes_np))

    def min(self, start: int = 0, end=None) -> float:
        """min(arr[start], ..., arr[end-1])"""
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        if start == 0 and end == self._capacity:
            return _lib.st_min_total(self._tree)
        return self._reduce(start, end - 1, 1, 0, self._capacity - 1)

    def _reduce(self, start, end, node, node_start, node_end):
        data = self._tree.contents.data
        if start == node_start and end == node_end:
            return data[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce(start, end, 2*node, node_start, mid)
        elif mid + 1 <= start:
            return self._reduce(start, end, 2*node+1, mid+1, node_end)
        else:
            return min(
                self._reduce(start, mid, 2*node, node_start, mid),
                self._reduce(mid+1, end, 2*node+1, mid+1, node_end)
            )
