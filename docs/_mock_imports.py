"""
Mock implementation for CutFEMx documentation generation.

This module provides mock classes and functions when the actual CutFEMx
C++ extensions are not available during documentation building.
"""

import sys
from unittest.mock import MagicMock


# Mock the C++ extensions that might not be available during docs build
class MockCppModule(MagicMock):
    """Mock C++ module for documentation generation."""

    def __getattr__(self, name):
        return "cpp." + name  # MagicMock()


# Mock cutfemx_cpp if not available
try:
    import cutfemx.cutfemx_cpp  # noqa: F401
except ImportError:
    sys.modules["cutfemx.cutfemx_cpp"] = MockCppModule()
    sys.modules["cutfemx.cutfemx_cpp.extensions"] = MockCppModule()
    sys.modules["cutfemx.cutfemx_cpp.quadrature"] = MockCppModule()

# Mock cutcells if not available
try:
    import cutcells  # noqa: F401
except ImportError:
    sys.modules["cutcells"] = MockCppModule()

# Mock dolfinx if not available
try:
    import dolfinx  # noqa: F401
except ImportError:
    sys.modules["dolfinx"] = MockCppModule()
    sys.modules["dolfinx.mesh"] = MockCppModule()
    sys.modules["dolfinx.fem"] = MockCppModule()
