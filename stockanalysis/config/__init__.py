"""配置层：PoolSpec 结构（schema）与加载 / 合并 / 校验（loader）。"""
from .loader import build_pool_spec, deep_merge, list_pools, load_global, load_pool, merged_config, parse_sets
from .schema import STATUSES, PoolSpec, RebalanceRule, Sizing

__all__ = ["PoolSpec", "Sizing", "RebalanceRule", "STATUSES", "load_pool", "list_pools", "load_global",
           "merged_config", "build_pool_spec", "deep_merge", "parse_sets"]
