"""Regenerate everything in README.md that is derived from code.

The README's ```python blocks are executed in order, in one namespace, like
notebook cells - so the code a reader sees is code that runs. Sections between
``<!-- generated:NAME -->`` and ``<!-- /generated -->`` are then rewritten from
what those blocks built, and the figures are drawn from the same objects. A
README edited by hand drifts from the code; one rebuilt from it cannot.

    python scripts/readme.py            rewrite the sections and redraw the figures
    python scripts/readme.py --check    exit 1 if a section is stale (no figures)

``tests/test_readme.py`` runs the check in CI. The benchmark tables are the one
exception: they need the hardware, so they are measured by the commands next
to them and stated with it.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import pathlib
import re
import sys
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ASSETS = ROOT / "assets"

_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_SECTION = re.compile(r"(<!-- generated:([\w-]+) -->\n)(.*?)(<!-- /generated -->)", re.DOTALL)


# ---- running the README ----------------------------------------------------------

def python_blocks(text: str) -> list:
    """The README's own python blocks - not those inside generated sections,
    which are output (the emitted kernel), not code to run."""
    authored = _SECTION.sub(lambda match: match.group(1) + match.group(4), text)
    return _BLOCK.findall(authored)


@functools.lru_cache(maxsize=None)
def run_blocks(text: str) -> dict:
    """Execute every python block in order; return the namespace they built.

    Once per process per text: a block may register a feature, and a backend
    rightly refuses a second feature under the same name.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    namespace = {"__name__": "readme"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for number, block in enumerate(python_blocks(text), 1):
            try:
                exec(compile(block, f"README.md python block {number}", "exec"), namespace)
            except Exception as error:
                raise RuntimeError(f"README python block {number} failed: {error!r}") from error
    return namespace


# ---- the generated sections ---------------------------------------------------------

def _fenced(body: str, language: str = "") -> str:
    return f"```{language}\n{body.rstrip()}\n```"


def _schedule(namespace) -> str:
    from kernelsmith.ir.explain import explain_graph
    return _fenced(explain_graph(namespace["g"]))


def _summary(namespace) -> str:
    from kernelsmith.ir import cse, fuse
    from kernelsmith.ir.fuse import FusedExpr
    g = namespace["g"]
    written = g.build()
    deduplicated, replace = cse(written)
    scheduled, _ = fuse(deduplicated, g.outputs.values(), replace)
    groups = [op for op in scheduled if isinstance(op, FusedExpr)]
    merged = sum(len(group.members) for group in groups)
    return (f"**{len(written)} operations were written; {len(scheduled)} run.** "
            f"CSE removed {len(written) - len(deduplicated)} duplicate call, and fusion merged "
            f"{merged} elementwise operations into {len(groups)} loops.")


def _cuda_source(namespace) -> str:
    # emission only: lowering and code generation need no GPU
    import kernelsmith.features  # noqa: F401  - registers the kernels
    from kernelsmith.backends.cuda import CudaBackend
    from kernelsmith.backends.generated import code_gen
    backend = CudaBackend()
    ops, layout = backend._lower(namespace["g"])
    return _fenced(code_gen(ops, layout, backend.registry), "python")


def _type_errors(namespace) -> str:
    from kernelsmith import Graph
    from kernelsmith.errors import KernelsmithError
    from kernelsmith.features import bootstrap_path

    g = Graph()
    close, opn = g.register_input("close"), g.register_input("open")
    history = g.register_table("history")
    attempts = {
        "(close + opn) & close": lambda: (close + opn) & close,
        "history * 2": lambda: history * 2,
        "bootstrap_path(close, 20, 0, 42)": lambda: bootstrap_path(close, 20, 0, 42),
    }
    lines = []
    for source, attempt in attempts.items():
        try:
            attempt()
        except KernelsmithError as error:
            lines += [f">>> {source}", f"{type(error).__name__}: {' '.join(str(error).split())}", ""]
        else:
            raise RuntimeError(f"expected '{source}' to raise")
    return _fenced("\n".join(lines))


def _features(namespace) -> str:
    from kernelsmith.dsl.graph import element
    from kernelsmith.features import indicators, specs

    from kernelsmith.features import kernels

    def signature(factory):
        args = ", ".join(repr(sig) for sig in factory.input_signature)
        outs = ", ".join(repr(sig) for sig in factory.output_signature)
        return f"`{factory.func_name}({args}) -> {outs}`"

    def summary(function):
        """The docstring's first sentence, on one line."""
        paragraph = " ".join((function.__doc__ or "").strip().split("\n\n")[0].split())
        sentence = re.split(r"(?<=\.)\s", paragraph)[0]
        return sentence.rstrip(".").replace("``", "`")

    rows = ["| | feature | what it computes |", "|---|---|---|"]
    rows += [f"| kernel | {signature(factory)} | {summary(getattr(kernels, factory.func_name + '_kernel'))} |"
             for factory in specs.FEATURES if factory is not element]
    rows += [f"| indicator | `{name}` | {summary(getattr(indicators, name))} |" for name in indicators.__all__]
    return "\n".join(rows)


SECTIONS = {
    "schedule": _schedule,
    "summary": _summary,
    "cuda-source": _cuda_source,
    "type-errors": _type_errors,
    "features": _features,
}


def render(text: str, namespace: dict) -> str:
    """The README with every generated section rebuilt."""
    def replace(match):
        name = match.group(2)
        if name not in SECTIONS:
            raise KeyError(f"README has an unknown generated section '{name}'")
        return f"{match.group(1)}{SECTIONS[name](namespace)}\n{match.group(4)}"
    return _SECTION.sub(replace, text)


# ---- the figures ---------------------------------------------------------------------

def draw_figures(namespace) -> None:
    from kernelsmith.ir import cse, fuse
    g = namespace["g"]
    ASSETS.mkdir(exist_ok=True)
    g.visualize(str(ASSETS / "example_graph.png"))
    scheduled, replace = cse(g.build())
    scheduled, levels = fuse(scheduled, g.outputs.values(), replace)
    g.visualize(str(ASSETS / "example_graph_compiled.png"), ops=scheduled, op_levels=levels, replace=replace)
    namespace["sim"].visualize(str(ASSETS / "simulation_graph.png"))

    spec = importlib.util.spec_from_file_location("option_pricing", ROOT / "examples" / "option_pricing.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.main(["--out-dir", str(ASSETS)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="only report stale sections")
    args = parser.parse_args(argv)

    text = README.read_text(encoding="utf-8")
    namespace = run_blocks(text)
    rendered = render(text, namespace)
    if args.check:
        if rendered != text:
            print("README.md is stale: run python scripts/readme.py", file=sys.stderr)
            return 1
        print("README.md is up to date")
        return 0
    README.write_text(rendered, encoding="utf-8")
    draw_figures(namespace)
    print("README.md and its figures regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
