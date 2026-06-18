# util/config.py
# -*- coding: utf-8 -*-

import os
import sys
import importlib.util


class Config:
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        self.cfg = self._load_py(cfg_path)

    def _load_py(self, path: str):
        assert os.path.isfile(path), f"Config file not found: {path}"
        spec = importlib.util.spec_from_file_location("cfg_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        # 把 module 里的变量导出来（过滤 __xxx__）
        cfg = {k: v for k, v in module.__dict__.items() if not k.startswith("__")}
        return cfg

    @staticmethod
    def _cli_keys() -> set:
        """
        解析命令行里显式传入的 key（--xxx 或 --xxx=...）
        用于：避免 config 覆盖 CLI
        """
        keys = set()
        for a in sys.argv[1:]:
            if a.startswith("--"):
                k = a[2:].split("=", 1)[0]
                keys.add(k)
        return keys

    def merge_to_args(self, args, allow_new_keys: bool = True, cli_has_priority: bool = True):
        """
        - allow_new_keys=True：config 里出现 argparse 没定义的 key 也不会 assert，直接 setattr 到 args
        - cli_has_priority=True：如果命令行显式传了 --k，则不让 config 覆盖该字段
        """
        cli_keys = self._cli_keys() if cli_has_priority else set()

        for k, v in self.cfg.items():
            if cli_has_priority and (k in cli_keys):
                # 命令行显式指定的参数，优先保留
                continue

            if hasattr(args, k):
                setattr(args, k, v)
            else:
                if allow_new_keys:
                    setattr(args, k, v)
                else:
                    raise AssertionError(f"Argument {k} is not defined")

        return args
