"""Small, read-only numerical API exposed to generated programs.

This is an in-process capability boundary, not an OS/container security claim.
In particular, importing NumPy must not grant file, process or native-pointer IO.
"""

from __future__ import annotations

import builtins
import importlib
from types import MappingProxyType


NUMERICAL_EXPORTS = {
    "numpy": frozenset({
        "array", "asarray", "ascontiguousarray", "copy", "zeros", "ones", "empty",
        "zeros_like", "ones_like", "full", "full_like", "eye", "identity",
        "arange", "linspace", "reshape", "ravel", "squeeze", "expand_dims",
        "stack", "vstack", "hstack", "concatenate", "split", "transpose", "swapaxes",
        "sum", "prod", "mean", "median", "quantile", "percentile", "std", "var",
        "min", "max", "amin", "amax", "argmin", "argmax", "argsort", "sort",
        "unique", "count_nonzero", "nonzero", "where", "clip", "abs", "absolute",
        "sqrt", "square", "power", "exp", "log", "log10", "sin", "cos", "tan",
        "arcsin", "arccos", "arctan", "arctan2", "degrees", "radians", "deg2rad",
        "rad2deg", "dot", "cross", "matmul", "einsum", "outer", "inner",
        "all", "any", "allclose", "isclose", "isfinite", "isnan", "isinf",
        "maximum", "minimum", "round", "floor", "ceil", "sign", "diff",
        "is_scalar", "isscalar", "ndim", "shape", "size", "diag", "diagonal",
        "triu", "tril", "take", "repeat", "tile", "broadcast_to",
        "float16", "float32", "float64", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "bool_", "integer", "floating",
        "number", "generic", "dtype", "ndarray", "pi", "e", "inf", "nan",
    }),
    "numpy.linalg": frozenset({
        "norm", "inv", "pinv", "solve", "lstsq", "det", "eig", "eigh", "eigvals",
        "eigvalsh", "svd", "qr", "matrix_rank", "cond",
    }),
    "scipy": frozenset(),
    "scipy.linalg": frozenset({
        "norm", "inv", "pinv", "solve", "lstsq", "det", "eig", "eigh", "svd", "qr",
    }),
    "scipy.spatial": frozenset({"ConvexHull", "KDTree", "cKDTree", "Delaunay"}),
    "scipy.spatial.distance": frozenset({
        "cdist", "pdist", "squareform", "euclidean", "cosine",
    }),
    "math": frozenset(name for name in dir(importlib.import_module("math"))
                      if not name.startswith("_")),
    "statistics": frozenset({
        "mean", "fmean", "geometric_mean", "harmonic_mean", "median", "median_low",
        "median_high", "mode", "multimode", "quantiles", "stdev", "pstdev",
        "variance", "pvariance", "covariance", "correlation",
    }),
}


class NumericalModule:
    __slots__ = ("_module_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_module_name", name)

    def __getattr__(self, name: str):
        module_name = object.__getattribute__(self, "_module_name")
        child = f"{module_name}.{name}"
        if child in NUMERICAL_EXPORTS:
            return NumericalModule(child)
        if name not in NUMERICAL_EXPORTS[module_name]:
            raise AttributeError(f"{module_name}.{name} is not a permitted numerical API")
        return getattr(importlib.import_module(module_name), name)

    def __setattr__(self, name, value):
        raise TypeError("numerical modules are read-only")


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in NUMERICAL_EXPORTS:
        raise ImportError(f"generated programs cannot import {name!r}")
    module = NumericalModule(name)
    for member in fromlist or ():
        if member == "*":
            raise ImportError("wildcard imports are not permitted")
        getattr(module, member)  # validate before Python binds the imported value
    return module if fromlist else NumericalModule(name.split(".")[0])


def safe_builtins():
    names = {
        "print", "len", "range", "float", "int", "str", "bool", "abs", "min", "max",
        "sum", "sorted", "list", "dict", "tuple", "set", "enumerate", "zip", "round",
        "isinstance", "format", "repr", "all", "any", "pow", "reversed", "next",
        "iter", "map", "filter", "slice",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "RuntimeError", "ArithmeticError", "ZeroDivisionError", "AssertionError",
    }
    return MappingProxyType({
        **{name: getattr(builtins, name) for name in names},
        "__import__": safe_import,
    })


def frozen_frames(frames):
    import numpy as np

    # A bytes-backed array cannot re-enable WRITEABLE, including via a view/base.
    return tuple(np.frombuffer(np.ascontiguousarray(p).tobytes(), dtype=p.dtype)
                 .reshape(p.shape) for p in frames)
