"""通用注册表：装饰器注册 + 参数名校验。

过滤规则（FILTERS）、打分器（SCORERS）、回测策略（STRATEGIES）共用本类。
新增一项 = 写函数/类 + 装饰器一行；池 yaml 中按名字引用，参数按关键字传入。

    FILTERS = Registry("filter", skip=1)       # 函数首参 ctx 不由 yaml 提供

    @FILTERS.register("boards", scope="entry")
    def boards(ctx, boards): ...

check_params() 用 inspect.signature 核对 yaml 给的参数：未知参数名、缺少必填参数
都在加载配置时报错，而不是等到运行中途。
"""
from __future__ import annotations

import inspect
from typing import Any, Callable


class Registry:
    def __init__(self, kind: str, skip: int = 0):
        """kind: 名称（报错提示用）；skip: 签名中前几个位置参数不由 yaml 提供。"""
        self.kind = kind
        self.skip = skip
        self._items: dict[str, Any] = {}
        self._meta: dict[str, dict] = {}

    def register(self, name: str, *, params: tuple[str, ...] | None = None, **meta) -> Callable:
        """装饰器。params 显式给出允许的参数名（签名含 **kwargs 时用），否则从签名推断。"""
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' 重复注册")
            self._items[name] = obj
            self._meta[name] = {"params": params, **meta}
            return obj
        return deco

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"未注册的 {self.kind}: '{name}'，可选: {self.names()}")
        return self._items[name]

    def meta(self, name: str) -> dict:
        self.get(name)
        return self._meta[name]

    def _signature_params(self, name: str) -> tuple[set[str], set[str], bool]:
        """返回 (全部参数名, 必填参数名, 是否接受任意关键字)。"""
        obj = self.get(name)
        explicit = self._meta[name]["params"]
        if explicit is not None:
            return set(explicit), set(), False
        fn = obj.__init__ if inspect.isclass(obj) else obj
        ps = list(inspect.signature(fn).parameters.values())
        if inspect.isclass(obj):
            ps = ps[1:]  # self
        ps = ps[self.skip:]
        names, required, var_kw = set(), set(), False
        for p in ps:
            if p.kind is p.VAR_KEYWORD:
                var_kw = True
            elif p.kind is p.VAR_POSITIONAL:
                continue
            else:
                names.add(p.name)
                if p.default is p.empty:
                    required.add(p.name)
        return names, required, var_kw

    def check_params(self, name: str, params: dict | None, where: str = "") -> None:
        params = params or {}
        names, required, var_kw = self._signature_params(name)
        loc = f"（{where}）" if where else ""
        unknown = set(params) - names
        if unknown and not var_kw:
            raise KeyError(f"{self.kind} '{name}' 不接受参数 {sorted(unknown)}{loc}，可用参数: {sorted(names)}")
        missing = required - set(params)
        if missing:
            raise KeyError(f"{self.kind} '{name}' 缺少必填参数 {sorted(missing)}{loc}")
