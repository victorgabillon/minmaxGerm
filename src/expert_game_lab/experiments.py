"""Command-line entry point and compatibility facade for experiment tooling.

The implementation is intentionally kept in :mod:`expert_game_lab.experiments_impl`.
Historically this module grew into a very large research workbench; keeping this
thin facade lets old imports and ``python -m expert_game_lab.experiments`` keep
working while we split the implementation into smaller modules over time.
"""

from __future__ import annotations

from . import experiments_impl as _impl


for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


def main() -> None:
    _impl.main()


if __name__ == "__main__":
    main()
