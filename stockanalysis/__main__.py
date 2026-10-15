"""python -X utf8 -m stockanalysis <命令>"""
import os
import sys

os.environ.setdefault("QLIB_KERNELS", "2")  # 本机提交内存上限约 32G；中证1000 训练时不宜更高

from .cli import main  # noqa: E402

sys.exit(main())
