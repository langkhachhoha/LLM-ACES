"""Expanded operator vocabulary used by every baseline (paper, Appendix B.2).

The paper states:

    All baselines except for APPS-ODE use the expanded operator set:
      Unary : sin, cos, tan, exp, log, sqrt, abs, tanh, sinh, cosh,
              square, cube, inv, neg, cbrt, log2, log10, exp2
      Binary: +, -, *, /, ^

Keeping the definition in one module guarantees SINDy / PySR / Operon / QBC / BO
all search the same hypothesis space, which is what the paper's "unified
experimental protocol" requires.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Canonical names (order matters: it is the order the paper lists them in)
# ---------------------------------------------------------------------------
UNARY_OPS: list[str] = [
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "sinh", "cosh",
    "square", "cube", "inv", "neg", "cbrt", "log2", "log10", "exp2",
]

BINARY_OPS: list[str] = ["+", "-", "*", "/", "^"]

# ---------------------------------------------------------------------------
# numpy callables + printable names, used to build the SINDy custom library
# ---------------------------------------------------------------------------
_NUMPY_UNARY = {
    "sin":    (np.sin,                     lambda s: f"sin({s})"),
    "cos":    (np.cos,                     lambda s: f"cos({s})"),
    "tan":    (np.tan,                     lambda s: f"tan({s})"),
    "exp":    (np.exp,                     lambda s: f"exp({s})"),
    "log":    (np.log,                     lambda s: f"log({s})"),
    "sqrt":   (np.sqrt,                    lambda s: f"sqrt({s})"),
    "abs":    (np.abs,                     lambda s: f"Abs({s})"),
    "tanh":   (np.tanh,                    lambda s: f"tanh({s})"),
    "sinh":   (np.sinh,                    lambda s: f"sinh({s})"),
    "cosh":   (np.cosh,                    lambda s: f"cosh({s})"),
    "square": (lambda x: x ** 2,           lambda s: f"({s})**2"),
    "cube":   (lambda x: x ** 3,           lambda s: f"({s})**3"),
    "inv":    (lambda x: 1.0 / x,          lambda s: f"1/({s})"),
    "neg":    (lambda x: -x,               lambda s: f"(-{s})"),
    "cbrt":   (np.cbrt,                    lambda s: f"({s})**(1/3)"),
    "log2":   (np.log2,                    lambda s: f"log({s})/log(2)"),
    "log10":  (np.log10,                   lambda s: f"log({s})/log(10)"),
    "exp2":   (np.exp2,                    lambda s: f"2**({s})"),
}


def numpy_unary_library(names: list[str] | None = None):
    """Return ``(functions, function_names)`` lists for pysindy.CustomLibrary."""
    names = names or UNARY_OPS
    fns, fn_names = [], []
    for n in names:
        f, printer = _NUMPY_UNARY[n]
        fns.append(f)
        fn_names.append(printer)
    return fns, fn_names


# ---------------------------------------------------------------------------
# PySR configuration (paper Table 10)
# ---------------------------------------------------------------------------
PYSR_UNARY = list(UNARY_OPS)
PYSR_BINARY = ["+", "-", "*", "/", "^"]

# Nested constraints prevent pathological compositions such as exp(exp(.)).
PYSR_NESTED_CONSTRAINTS = {
    "exp":  {"exp": 0, "log": 0, "exp2": 0, "log2": 0, "log10": 0},
    "exp2": {"exp": 0, "log": 0, "exp2": 0, "log2": 0, "log10": 0},
    "log":  {"exp": 0, "log": 0, "exp2": 0, "log2": 0, "log10": 0},
    "log2": {"exp": 0, "log": 0, "exp2": 0, "log2": 0, "log10": 0},
    "log10": {"exp": 0, "log": 0, "exp2": 0, "log2": 0, "log10": 0},
    "sin":  {"sin": 0, "cos": 0, "tan": 0},
    "cos":  {"sin": 0, "cos": 0, "tan": 0},
    "tan":  {"sin": 0, "cos": 0, "tan": 0},
    "sqrt": {"sqrt": 0, "cbrt": 0},
    "cbrt": {"sqrt": 0, "cbrt": 0},
    "tanh": {"tanh": 0, "sinh": 0, "cosh": 0},
    "sinh": {"tanh": 0, "sinh": 0, "cosh": 0},
    "cosh": {"tanh": 0, "sinh": 0, "cosh": 0},
    "inv":  {"inv": 0},
    "neg":  {"neg": 0},
}

# Restrict exponentiation ranges so that x^y stays tractable.
PYSR_CONSTRAINTS = {"^": (-1, 1)}


# ---------------------------------------------------------------------------
# Operon configuration (paper Table 10)
# ---------------------------------------------------------------------------
OPERON_SYMBOLS = (
    "add,sub,mul,div,aq,pow,abs,cbrt,cos,cosh,exp,log,sin,sinh,sqrt,tan,tanh,"
    "square,constant,variable"
)


# ---------------------------------------------------------------------------
# APPS-ODE grammar function set (paper, Appendix B.2 -- deliberately smaller)
# ---------------------------------------------------------------------------
APPS_ODE_FUNCTION_SET = ["add", "sub", "mul", "div", "sin", "exp", "poly", "const"]
