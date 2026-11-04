"""Latent-trait compatibility API."""
from .factor_gwas_api import factorGWAS

def factor_gwas(*args,**kwargs):
    """Return the factor association result computed by ``factorGWAS``.

    Parameters
    ----------
    *args : tuple
        Additional positional arguments forwarded, in order, to ``factorGWAS``. See that
        signature for required scientific inputs and their order.

    **kwargs : mapping
        Additional keyword arguments forwarded to ``factorGWAS``. See those callables for
        accepted keys and values. Passed to ``factorGWAS`` as ``**expanded mapping``.

    Returns
    -------
    object
        Computed result from ``factorGWAS(*args, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    kwargs.setdefault('loco_factor','refit')
    kwargs.setdefault('method','exact')
    return factorGWAS(*args,**kwargs)
__all__=['factor_gwas','factorGWAS']
