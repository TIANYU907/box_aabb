"""Build optional v3 Cython extensions.

Usage:
    python v3/setup_cython.py build_ext --inplace
"""

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "aabb._fk_scalar_core",
        sources=["src/aabb/_fk_scalar_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "aabb._interval_fk_core",
        sources=["src/aabb/_interval_fk_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "forest._hier_core",
        sources=["src/forest/_hier_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)
