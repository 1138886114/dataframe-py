from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# 通用编译器优化标志
COMMON_compile_args = [
    "-O3",                   # 最高优化级别
    "-march=native",         # 针对本机 CPU 优化 (ARM64 NEON)
    "-ffast-math",           # 快速浮点运算 (牺牲极小精度换速度)
    "-flto",                 # 链接时优化 (Link-Time Optimization)
    "-funroll-loops",        # 循环展开
    "-fno-math-errno",       # 跳过 math 函数错误检查
]

# 链接器优化
COMMON_link_args = ["-flto"]

# C++ 模块额外标志
CXX_compile_args = COMMON_compile_args + ["-std=c++17"]

extensions = [
    Extension("fluidframe.core._base",
              ["fluidframe/core/_base.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.intcolumn",
              ["fluidframe/core/intcolumn.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.doublecolumn",
              ["fluidframe/core/doublecolumn.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.char256column",
              ["fluidframe/core/char256column.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.ucs2column",
              ["fluidframe/core/ucs2column.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.datecolumn",
              ["fluidframe/core/datecolumn.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.boolcolumn",
              ["fluidframe/core/boolcolumn.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.categorycolumn",
              ["fluidframe/core/categorycolumn.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.datetimecolumn",
              ["fluidframe/core/datetimecolumn.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.timecolumn",
              ["fluidframe/core/timecolumn.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.index",
              ["fluidframe/core/index.pyx"],
              language="c++",
              extra_compile_args=CXX_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.indexers",
              ["fluidframe/core/indexers.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.fastdataframe",
              ["fluidframe/core/fastdataframe.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.groupby",
              ["fluidframe/core/groupby.pyx"],
              language="c++",
              extra_compile_args=CXX_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.linalg",
              ["fluidframe/core/linalg.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.ucs2_str_accessor",
              ["fluidframe/core/ucs2_str_accessor.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.char256_str_accessor",
              ["fluidframe/core/char256_str_accessor.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core.fluidframe",
              ["fluidframe/core/fluidframe.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
]

setup(
    name="fluidframe",
    packages=["fluidframe", "fluidframe.core"],
    ext_modules=cythonize(extensions, language_level=3),
)
