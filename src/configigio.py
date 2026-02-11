"""Backward-compat shim for typo'd module name `configigio`.

Some installations may still import `configigio`; re-export symbols from
`configio` so older entrypoints keep working.
"""

from configio import DEFAULT_CONFIG, load_config, save_config

__all__ = ["DEFAULT_CONFIG", "load_config", "save_config"]
