"""Compatibility wrapper for the moved parse-and-label utilities."""

from scripts.data import parse_and_label as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

__all__ = [_name for _name in globals() if not _name.startswith("__")]
