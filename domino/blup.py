"""Deprecated domino-blup command and legacy BLUP import facade."""
import warnings
import vigor.blup as _canonical
from vigor.blup import *  # noqa: F403
from vigor.blup import main as _main


def __getattr__(name):
    """Resolve a legacy facade attribute from the canonical VIGOR module.

    Parameters
    ----------
    name : str
        Identifier used to select or label the relevant component, field or output quantity. It
        must match the names present in the supplied object where lookup is required.
    """
    return getattr(_canonical, name)


def main(argv=None):
    """Parse command-line options and execute the requested VIGOR workflow.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line argument tokens. None reads the process argument vector; accepted switches
        and choices are defined by the corresponding parser. Default: ``None``.

    Returns
    -------
    object
        Computed result from ``_main(argv)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    warnings.warn('The domino-blup command is deprecated; use vigor-blup.', FutureWarning, stacklevel=2)
    return _main(argv)


if __name__ == '__main__':
    main()
