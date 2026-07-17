# dataframe-py

#### 介绍
自定义数据处理库
- 固定字段处理库 SolidDataFrame
- 注册字段处理库 FluidDataFrame

这两个库是相互独立的：
- SolidDataFrame 专门处理OCR数据的
- FluidDataFrame 专门处理已知数据结构与字段类型的数据
- 与pandas的区别：SolidDataFrame是固定字段，FluidDataFrame是用户临时注册字段，而pandas是自由字段，在使用pandas处理数据时，可以不用知道数据的字段个数与字段的数据类型，特别自由。


#### 安装或者使用教程
将代码放置在使用代码的同级目录即可，执行如下命令：
```shell
python3 -c "from setuptools import setup; from Cython.Build import cythonize; setup(ext_modules=cythonize('ndframe.pyx', language_level=3))" build_ext --build-lib=build
```
