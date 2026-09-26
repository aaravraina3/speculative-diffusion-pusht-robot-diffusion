"""Python 3.14 shims for lerobot 0.4.x. Import this before lerobot.

draccus hands argparse type annotations like ``Dict[str, PolicyFeature] | None``
as ``type=``, which Python 3.14's argparse rejects as not callable. pyarrow raises
if pandas re-registers an extension type that lerobot's import already registered.
Both are harmless to paper over for inference. On Python 3.12 neither shim runs
anything, so the clean fix is a 3.12 environment (see pyproject.toml).
"""
import argparse
import sys

import pyarrow as pa
import pyarrow.lib as palib


def _patch_argparse() -> None:
    def patch(cls):
        if getattr(cls.add_argument, "_lerobot_patched", False):
            return
        orig = cls.add_argument

        def add_argument(self, *args, **kwargs):
            t = kwargs.get("type")
            if t is not None and not callable(t):
                kwargs["type"] = lambda v: v
            return orig(self, *args, **kwargs)

        add_argument._lerobot_patched = True
        cls.add_argument = add_argument

    patch(argparse.ArgumentParser)
    patch(argparse._ArgumentGroup)


def _patch_pyarrow() -> None:
    if getattr(palib.register_extension_type, "_safe_patched", False):
        return
    register, unregister = palib.register_extension_type, palib.unregister_extension_type

    def safe_register(ext):
        try:
            return register(ext)
        except pa.lib.ArrowKeyError:  # already registered by an earlier import
            return None

    def safe_unregister(name):
        try:
            return unregister(name)
        except pa.lib.ArrowKeyError:
            return None

    safe_register._safe_patched = True
    palib.register_extension_type = pa.register_extension_type = safe_register
    palib.unregister_extension_type = pa.unregister_extension_type = safe_unregister
    for mod in list(sys.modules):
        if mod.startswith(("pandas.core.arrays.arrow", "pandas.io.parquet")):
            del sys.modules[mod]


if sys.version_info >= (3, 14):
    _patch_argparse()
    _patch_pyarrow()
