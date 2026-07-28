"""Exception hierarchy for kernelsmith.

Everything raised by this package derives from KernelsmithError, so callers
can catch one type. DSL misuse fails loudly at graph-build time, never at
codegen or run time.
"""


class KernelsmithError(Exception):
    """Base class for all kernelsmith errors."""


class DslTypeError(KernelsmithError):
    """Operand or argument types do not satisfy an operator or feature signature."""


class GraphError(KernelsmithError):
    """Invalid graph construction: duplicate names, wrong containers, empty graphs."""
