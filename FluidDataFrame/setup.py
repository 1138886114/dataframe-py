from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("fluidframe.core._base", ["fluidframe/core/_base.pyx"]),
    Extension("fluidframe.core.intcolumn", ["fluidframe/core/intcolumn.pyx"]),
    Extension("fluidframe.core.doublecolumn", ["fluidframe/core/doublecolumn.pyx"]),
    Extension("fluidframe.core.char256column", ["fluidframe/core/char256column.pyx"]),
    Extension("fluidframe.core.ucs2column", ["fluidframe/core/ucs2column.pyx"]),
    Extension("fluidframe.core.timestampcolumn", ["fluidframe/core/timestampcolumn.pyx"]),
    Extension("fluidframe.core.boolcolumn", ["fluidframe/core/boolcolumn.pyx"]),
    Extension("fluidframe.core.categorycolumn", ["fluidframe/core/categorycolumn.pyx"]),
    Extension("fluidframe.core.datetimecolumn", ["fluidframe/core/datetimecolumn.pyx"]),
    Extension("fluidframe.core.timecolumn", ["fluidframe/core/timecolumn.pyx"]),
    Extension("fluidframe.core.index", ["fluidframe/core/index.pyx"], language="c++"),
    Extension("fluidframe.core.indexers", ["fluidframe/core/indexers.pyx"]),
    Extension("fluidframe.core.fastdataframe", ["fluidframe/core/fastdataframe.pyx"]),
    Extension("fluidframe.core.groupby", ["fluidframe/core/groupby.pyx"], language="c++"),
    Extension("fluidframe.core.linalg", ["fluidframe/core/linalg.pyx"]),
    Extension("fluidframe.core.fluidframe", ["fluidframe/core/fluidframe.pyx"]),
]

setup(
    name="fluidframe",
    packages=["fluidframe", "fluidframe.core"],
    ext_modules=cythonize(extensions, language_level=3),
)
