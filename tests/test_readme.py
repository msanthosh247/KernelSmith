"""The README is built from the code, and stays that way.

Every python block in it runs, in order, and every generated section - the
compiler's report, the emitted kernel, the feature table, the error messages -
must equal what ``scripts/readme.py`` produces now. If this fails, run:

    python scripts/readme.py
"""
import importlib.util
import pathlib

import pytest

pytest.importorskip("numba")

ROOT = pathlib.Path(__file__).parent.parent


def load_generator():
    spec = importlib.util.spec_from_file_location("readme_generator", ROOT / "scripts" / "readme.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built():
    generator = load_generator()
    text = generator.README.read_text(encoding="utf-8")
    return generator, text, generator.run_blocks(text)


def test_every_python_block_runs(built):
    generator, text, namespace = built
    assert len(generator.python_blocks(text)) >= 3
    assert namespace["out"]["signal"].shape == (3, 250)


def test_generated_sections_are_current(built):
    from kernelsmith.availability import cuda_importable, cuda_unavailable_reason
    if not cuda_importable():
        # the README shows the emitted CUDA kernel; emitting it needs numba.cuda.
        # CI checks this under the simulator, where numba.cuda always imports.
        pytest.skip(f"cannot emit the CUDA section here: {cuda_unavailable_reason}")
    generator, text, namespace = built
    assert generator.render(text, namespace) == text, "README is stale: run python scripts/readme.py"


def test_every_figure_the_readme_shows_exists():
    import re
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    local = [path for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) if "://" not in path]
    assert local
    for path in local:
        assert (ROOT / path).is_file(), path
