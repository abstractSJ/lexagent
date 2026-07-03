"""
测试包入口。

保留这个文件的原因，是让 `python -m unittest discover` 能稳定把 `tests/` 目录识别为可导入包，
避免不同运行方式下出现 `ModuleNotFoundError: No module named 'tests.xxx'`。
"""
