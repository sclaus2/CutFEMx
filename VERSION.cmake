# CutFEMx Version Information
# This file is the single source of truth for version numbers

set(CUTFEMX_VERSION_MAJOR 0)
set(CUTFEMX_VERSION_MINOR 1)
set(CUTFEMX_VERSION_PATCH 0)
set(CUTFEMX_VERSION "${CUTFEMX_VERSION_MAJOR}.${CUTFEMX_VERSION_MINOR}.${CUTFEMX_VERSION_PATCH}")

# Optional: Pre-release or build metadata
set(CUTFEMX_VERSION_SUFFIX "")  # e.g., "-dev", "-alpha1", "-rc1"
if(CUTFEMX_VERSION_SUFFIX)
    set(CUTFEMX_VERSION_FULL "${CUTFEMX_VERSION}${CUTFEMX_VERSION_SUFFIX}")
else()
    set(CUTFEMX_VERSION_FULL "${CUTFEMX_VERSION}")
endif()