"""测试全局前置。

只干一件事：把运行数据目录指到临时目录。

app/main.py 一被 import 就会建账号库 —— 任何 import 了它的测试（哪怕
只想拿一个纯函数）都会在开发机真实的 data/ 里写出 users.json。跑一次
测试就把自己的账号覆盖掉，这种事必须在 collection 之前就堵住，
而 conftest.py 正是唯一比测试模块 import 更早执行的地方。
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("LCM90_DATA_DIR",
                      tempfile.mkdtemp(prefix="lcm90-test-data-"))
