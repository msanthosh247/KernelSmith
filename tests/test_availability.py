"""numba-cuda installed without a CUDA runtime must not break the package.

Recent numba-cuda loads libcudart while numba.cuda is imported, and raises -
not an ImportError - when it is missing. That is every CPU-only machine with
numba-cuda installed, CI runners included. Each case runs in a fresh process:
the probe's answer is cached, and numba.cuda may already be imported here.
"""
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("numba")

NO_RUNTIME = textwrap.dedent('''
    import sys

    class NoCudaRuntime:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "numba.cuda.device_init":
                raise RuntimeError("Failure finding libcudart (simulated)")
            return None

    sys.meta_path.insert(0, NoCudaRuntime())
''')


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", NO_RUNTIME + textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=300)


def test_features_import_and_cpu_backends_work_without_a_cuda_runtime():
    result = run('''
        import numpy as np
        from kernelsmith import Graph, availability
        from kernelsmith.features import sma
        from kernelsmith.backends.numba_cpu import NumbaCPU_Backend

        assert not availability.cuda_importable()
        assert "libcudart" in availability.cuda_unavailable_reason
        assert not availability.cuda_usable()

        g = Graph()
        close = g.register_input("close")
        g.register_output("mean", sma(close, 3))
        out = NumbaCPU_Backend().compile(g).run({"close": np.arange(5, dtype=np.float32)}, {})
        assert out["mean"][0, -1] == 3.0
        print("ok")
    ''')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_probe_reports_a_missing_package():
    result = run('''
        import sys
        sys.modules["numba_cuda"] = None          # as if not installed
        import importlib.util
        real = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *a: None if name == "numba_cuda" else real(name, *a)
        from kernelsmith import availability
        assert not availability.cuda_importable()
        print(availability.cuda_unavailable_reason)
    ''')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "numba-cuda is not installed"
