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

extensions = [
    Extension("fluidframe.core._columns",
              ["fluidframe/core/_columns.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
    Extension("fluidframe.core._databaseframe",
              ["fluidframe/core/_databaseframe.pyx"],
              extra_compile_args=COMMON_compile_args,
              extra_link_args=COMMON_link_args),
]

setup(
    name="fluidframe",
    packages=["fluidframe", "fluidframe.core"],
    ext_modules=cythonize(extensions, language_level=3),
)
