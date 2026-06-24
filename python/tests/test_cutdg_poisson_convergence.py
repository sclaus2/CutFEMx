# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cutdg_poisson_manufactured_solution_converges():
    demo_path = Path(__file__).parents[1] / "demo" / "demo_dg_poisson.py"
    demo_args = [
        str(demo_path),
        "--n",
        "4",
        "--levels",
        "2",
        "--order",
        "3",
        "--check-convergence",
        "--min-rate",
        "1.0",
    ]
    code = (
        "import os, runpy, sys; "
        f"sys.path.insert(0, {str(demo_path.parent)!r}); "
        f"sys.argv = {demo_args!r}; "
        f"runpy.run_path({str(demo_path)!r}, run_name='__main__'); "
        "sys.stdout.flush(); "
        "os._exit(0)"
    )
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
