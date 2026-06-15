"""Pytest session setup shared by all tests in tests/.

Sets KMP_DUPLICATE_LIB_OK before torch is imported. On macOS, torch and
numpy/MKL each bundle their own libomp, and the duplicate OpenMP runtime
aborts the process at import time with "OMP: Error #15". Allowing the
duplicate runtime is the standard workaround for a test run. This must run
before any test module imports torch, so it lives at import time in conftest.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
