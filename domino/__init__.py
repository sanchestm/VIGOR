"""Temporary Domino import facade. New code should import :mod:`vigor`."""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import vigor as _canonical


class _AliasLoader(importlib.abc.Loader):
    """_AliasLoader represents alias loader.

    See ``__init__`` for complete input types, defaults and construction constraints.
    """
    def __init__(self, canonical_name):
        """Initialize _AliasLoader with the inputs and resource settings below.

        Parameters
        ----------
        canonical_name : str, module, import specification or path-like
            Import/resource selector consumed by the explicit operation. In the Domino import facade
            these follow importlib's module-name, spec and search-path protocol; file I/O functions
            use the path format named in their summary.
        """
        self.canonical_name = canonical_name

    def create_module(self, spec):
        """Return the canonical implementation module for a legacy Domino import.

        Parameters
        ----------
        spec : covariance specification or importlib.ModuleSpec
            Prepared matrix representation for numerical helpers, or importlib specification in the
            Domino import facade. Use the object supplied by the corresponding normalizer/import
            protocol, with unchanged shape/module identity.

        Returns
        -------
        object
            Computed result from ``self.module``; see the summary and the called implementation for its labeled/numerical representation.
        """
        self.module = importlib.import_module(self.canonical_name)
        self.canonical_spec = self.module.__spec__
        return self.module

    def exec_module(self, module):
        # Import machinery assigns the alias spec to existing modules. Restore
        # the canonical spec so reload, introspection and pickles use vigor.
        """Keep the legacy alias bound to the already-executed canonical module.

        Parameters
        ----------
        module : str, module, import specification or path-like
            Import/resource selector consumed by the explicit operation. In the Domino import facade
            these follow importlib's module-name, spec and search-path protocol; file I/O functions
            use the path format named in their summary.
        """
        module.__spec__ = self.canonical_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """_AliasFinder represents alias finder.
    """
    def find_spec(self, fullname, path=None, target=None):
        """Resolve legacy Domino submodule imports to their canonical VIGOR implementation.

        Parameters
        ----------
        fullname : str
            Fully qualified import name. Names outside domino.* and the physical
            wrappers domino.__main__, domino.cli and domino.blup are left to
            ordinary import discovery. Other legacy names resolve to vigor.*.

        path : sequence of str or None
            Default: ``None``. Parent-package search locations supplied by
            importlib; unused because canonical lookup uses the qualified name.

        target : module or None
            Default: ``None``. Existing reload target supplied by importlib;
            unused by this alias finder.

        Returns
        -------
        importlib.machinery.ModuleSpec or None
            Alias-loader specification preserving canonical module identity and
            package status, or None for physical wrappers/missing modules.

        Complexity: O(l)+C_lookup TIME and O(l)+S_lookup SPACE, where l is module
        name length and C_lookup/S_lookup are canonical importlib discovery
        work/storage (which can import parent packages). No scientific arrays
        or data files are loaded by this finder itself.
        """
        if not fullname.startswith('domino.') or fullname in {'domino.__main__', 'domino.cli', 'domino.blup'}:
            return None
        canonical_name = 'vigor' + fullname[len('domino'):]
        spec = importlib.util.find_spec(canonical_name)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(canonical_name),
            is_package=spec.submodule_search_locations is not None,
        )


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

# Alias modules already loaded by vigor, preserving all function globals and
# circular-import identities. Unloaded optional modules use the finder above.
for _name, _module in list(sys.modules.items()):
    if _name.startswith('vigor.') and _name not in {'vigor.__main__', 'vigor.cli', 'vigor.blup'}:
        sys.modules['domino' + _name[len('vigor'):]] = _module

__version__ = _canonical.__version__
__all__ = _canonical.__all__
globals().update({name: getattr(_canonical, name) for name in __all__})


def __getattr__(name):
    """Resolve a legacy facade attribute from the canonical VIGOR module.

    Parameters
    ----------
    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.
    """
    return getattr(_canonical, name)


warnings.warn('Domino is now VIGOR; use import vigor. The domino facade is temporary.',
              DeprecationWarning, stacklevel=2)
