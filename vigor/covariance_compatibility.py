"""Strict, evidence-backed compatibility dispatch for covariance estimation."""
from __future__ import annotations

from collections.abc import Mapping
from functools import wraps


_GDEC314_ALLOWED = {
    "mode", "block_size", "block_dtype", "accum_dtype", "square_dtype",
    "dense_threshold_gb", "rank_G", "rank_E", "psd_E", "ridge_G",
    "ridge_E", "method", "maxiter", "gtol", "ftol", "init", "eps",
    "n_boot", "random_state", "shrink_G", "shrink_E", "E_structure",
    "min_E_pair_overlap", "blas_threads",
}

_GDEC314_DEFAULTS = {
    "rank_G": None,
    "rank_E": None,
    "shrink_G": "auto",
    "shrink_E": "auto",
    "E_structure": "auto",
    "method": "L-BFGS-B",
    "maxiter": 1000,
    "gtol": 1e-8,
    "ftol": 1e-12,
    "eps": 1e-12,
    "n_boot": 0,
}


def covariance_compatibility(function):
    """Decorate covariance fitting with the pinned GDEC 3.1.4 compatibility route.
    
    Parameters
    ----------
    function : callable
        Current covariance estimator accepting grm,traits and keyword options.
        Called unchanged when the wrapper receives compatibility=None.
    
    Returns
    -------
    callable
        functools.wraps-preserved wrapper. compatibility='gdec314' uses the captured
        _ORIGINAL_LEGACY_GENETIC_VARCOV, aligns kernel labels to traits, and annotates
        the returned Series. It does not invoke the supplied callable in that mode.
    
    Notes
    -----
    GDEC 3.1.3 is explicitly unavailable in this route; genuine 3.1.3 comparisons
    must load the separately pinned comparator. Bootstrap/CV/multiple kernels are
    unsupported for this compatibility profile.
    
    Complexity
    ----------
    Decoration TIME O(1), SPACE O(1) references. Invocation costs are documented on
    wrapped: arbitrary callback/estimator work is additional, never O(1).
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        """Dispatch current fitting or explicitly requested GDEC 3.1.4 compatibility.
        
        Parameters
        ----------
        *args : tuple
            Positional estimator inputs, beginning with grm,traits when positional.
        **kwargs : dict
            compatibility=None delegates all remaining keys to the decorated function.
            'gdec314' (case/_/- normalized) permits _GDEC314_ALLOWED covariance options,
            grm and traits; progress is consumed. Defaults are rank_G/rank_E=None,
            shrink_G/shrink_E/E_structure='auto', method='L-BFGS-B', maxiter=1000,
            gtol=1e-8,ftol=1e-12,eps=1e-12,n_boot=0. optimize_hyperparameters_cv must be 0.
        
        Returns
        -------
        object
            Unmodified callback result for compatibility=None. Historical route returns
            its covariance Series with compatibility='gdec314', zero optimization,
            hyperparameter_optimization=None, compatibility_engine and an explicit
            compatibility_profile of resolved default/overridden estimator options.
        
        Raises
        ------
        ValueError
            Unknown profile or incompatible enabled hyperparameter CV.
        NotImplementedError
            gdec313, mapping GRM or nonzero bootstrap count requested.
        TypeError
            Missing grm or an unsupported compatibility keyword.
        
        Complexity
        ----------
        TIME O(k+l) wrapper work plus delegated estimator/alignment costs; SPACE O(k+l)
        plus those costs, k keyword count, l profile string length. Label alignment may
        reorder/copy an n-by-n dense kernel (O(n²) time/space). For arbitrary decorated
        functions no universal estimator bound exists; compatibility delegates to
        multivariate._ORIGINAL_LEGACY_GENETIC_VARCOV with the resolved model/ranks.
        """
        compatibility = kwargs.pop("compatibility", None)
        if compatibility is None:
            return function(*args, **kwargs)

        profile = str(compatibility).lower().replace("_", "").replace("-", "")
        if profile == "gdec313":
            raise NotImplementedError(
                "compatibility='gdec313' is unavailable because no pinned "
                "gdec 3.1.3 reference implementation was supplied"
            )
        if profile != "gdec314":
            raise ValueError("compatibility must be None or 'gdec314'")

        from .hyperparameter_cv import validate_cv_count

        cv = validate_cv_count(kwargs.pop("optimize_hyperparameters_cv", 0))
        if cv:
            raise ValueError("optimize_hyperparameters_cv is unavailable with compatibility='gdec314'; use the standard covariance path")

        if len(args) >= 1:
            grm = args[0]
        elif "grm" in kwargs:
            grm = kwargs["grm"]
        else:
            raise TypeError("missing required argument: 'grm'")
        if isinstance(grm, Mapping):
            raise NotImplementedError(
                "compatibility='gdec314' is currently validated only for a single GRM"
            )

        progress = kwargs.pop("_runtime_progress", None)
        # ``progress`` is owned by the outer runtime decorator and is not a
        # historical estimator argument.
        kwargs.pop("progress", None)
        unknown = sorted(set(kwargs) - _GDEC314_ALLOWED - {"grm", "traits"})
        if unknown:
            raise TypeError(
                "unsupported argument(s) for compatibility='gdec314': "
                + ", ".join(unknown)
            )

        historical_kwargs = dict(_GDEC314_DEFAULTS)
        historical_kwargs.update(kwargs)
        if int(historical_kwargs.get("n_boot", 0)) != 0:
            raise NotImplementedError(
                "compatibility='gdec314' currently requires n_boot=0; bootstrap "
                "parity has not been validated"
            )
        if progress is not None:
            progress.stage("Covariance: gdec 3.1.4 compatibility engine")

        from .multivariate import _ORIGINAL_LEGACY_GENETIC_VARCOV, _align_labeled_kernels_to_traits

        traits = args[1] if len(args) > 1 else historical_kwargs.get("traits")
        aligned = _align_labeled_kernels_to_traits(grm, traits)
        if args:
            args = (aligned, *args[1:])
        else:
            historical_kwargs["grm"] = aligned

        result = _ORIGINAL_LEGACY_GENETIC_VARCOV(*args, **historical_kwargs)
        result["compatibility"] = "gdec314"
        result["optimize_hyperparameters_cv"] = 0
        result["hyperparameter_optimization"] = None
        result["compatibility_engine"] = "captured_gdec314_covariance"
        result["compatibility_profile"] = {
            key: historical_kwargs[key]
            for key in ("rank_G", "rank_E", "shrink_G", "shrink_E", "E_structure",
                        "method", "maxiter", "gtol", "ftol", "eps", "n_boot")
        }
        return result

    return wrapped
