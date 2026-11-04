"""Explicit adapter for legacy project threshold files.

The historical PVALTHRESHOLD.csv stores -log10(p), not raw p. The adapter
never treats an unprovenanced historical file as an automatic cache hit.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .effective_tests import effective_tests, empirical_threshold_to_meff


def read_legacy_thresholds(path):
    """Read alpha-indexed legacy ``thresholds`` values as -log10(p).

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    Returns
    -------
    object
        Computed result from ``pd.DataFrame(rows).set_index('alpha')``; see the summary and the called implementation for its labeled/numerical representation.
    """
    path = Path(path)
    if path.is_dir():
        path = path / 'pvalthresh' / 'PVALTHRESHOLD.csv'
    frame = pd.read_csv(path, index_col=0)
    if 'thresholds' not in frame:
        raise ValueError("legacy threshold CSV requires a 'thresholds' column")
    rows=[]
    for label, value in frame['thresholds'].items():
        label=str(label).strip()
        if not label.endswith('%'):
            continue  # legacy pandas.describe also saved count, mean, min, etc.
        alpha=float(label[:-1])/100
        value=float(value)
        if not 0 < alpha < 1 or not np.isfinite(value) or value <= 0:
            raise ValueError(f'invalid legacy threshold at {label!r}')
        p=10.0**(-value)
        if p == 0:
            raise ValueError('legacy threshold underflows raw p representation')
        rows.append({'alpha':alpha,'p_threshold':p,'neglog10_threshold':value,
                     'empirical_equivalent_meff':empirical_threshold_to_meff(p,alpha)})
    if not rows:
        raise ValueError('legacy threshold CSV contains no percentage-indexed thresholds')
    return pd.DataFrame(rows).set_index('alpha')


class EffectiveTestsMixin:
    """Opt-in threshold methods for an existing project object.

    Required attributes: ``genotypes_subset`` and optionally ``path``.
    The methods do not replace the object's GWAS implementation.
    """
    def effective_tests(self, **kwargs):
        """Estimate effective GWAS tests using the threshold object's configured genotype source.

        Parameters
        ----------
        **kwargs : mapping
            Additional keyword arguments forwarded to ``effective_tests``. See those callables for
            accepted keys and values. Passed to ``effective_tests`` as ``**expanded mapping``.

        Returns
        -------
        object
            Computed result from ``effective_tests(source, **kwargs)``; see the summary and the called implementation for its labeled/numerical representation.
        """
        source=self.genotypes_subset
        if isinstance(source, (str, Path)):
            from ..genotype_source import open_genotypes
            source=open_genotypes(source)
            try:
                return effective_tests(source, **kwargs)
            finally:
                if hasattr(source,'close'):
                    source.close()
        return effective_tests(source, **kwargs)

    def estimate_pval_threshold(self, method='auto', *, alpha=(.10,.05),
                                allow_legacy_cache=False, **kwargs):
        """Estimate and save legacy-compatible -log10 thresholds.

        ``method='legacy_null_gcta'`` requires a user-supplied legacy callback.
        No historical threshold is reused without explicit permission because
        old CSVs do not identify genotype QC, sample set, or fitted null model.

        Parameters
        ----------
        method : callable or str
            Algorithm/estimator selected for the operation described above. See the source-derived
            choices and called model factory for the accepted methods; callable methods must obey
            that factory's fit/predict or objective contract. Default: ``'auto'``. Source-recognized
            literal values: ``legacy_null_gcta``. These select the corresponding branches below;
            comparison evidence alone does not establish an exhaustive enum. Passed to
            ``self.effective_tests`` as ``method``.

        alpha : float
            Regularization coefficient, mixing weight or statistical significance threshold selected
            by this estimator; the local validation and objective determine its precise scale.
            Default: ``(0.1, 0.05)``. Passed to ``self.effective_tests`` as ``alpha``.

        allow_legacy_cache : bool
            Context/control value read by the enclosing operation; its contract follows the explicit
            local use and the caller. Default: ``False``.

        **kwargs : mapping
            Additional keyword arguments forwarded to ``callback``, ``self.effective_tests``. See
            those callables for accepted keys and values. Passed to ``callback`` as ``**expanded
            mapping``.

        Returns
        -------
        object
            Computed result from ``table``; see the summary and the called implementation for its labeled/numerical representation.
        """
        if method == 'legacy_null_gcta':
            callback=getattr(self,'legacy_threshold_estimator',None)
            if callback is None:
                raise NotImplementedError('provide legacy_threshold_estimator to run the external GCTA reference')
            return callback(**kwargs)
        path=Path(getattr(self,'path','.'))/'pvalthresh'/'PVALTHRESHOLD.csv'
        if method == 'auto' and allow_legacy_cache and path.exists():
            warnings.warn('Reusing legacy thresholds without sample/model provenance; caller must verify compatibility.',
                          RuntimeWarning,stacklevel=2)
            saved=read_legacy_thresholds(path)
            values={float(a):float(saved.loc[float(a),'neglog10_threshold']) for a in np.atleast_1d(alpha)}
            self.effective_tests_result=None
        else:
            result=self.effective_tests(method=method,alpha=alpha,**kwargs)
            if not result.neglog10_threshold:
                raise ValueError('this diagnostic result does not define a GWAS significance threshold')
            values={float(a):float(v) for a,v in result.neglog10_threshold.items()}
            self.effective_tests_result=result
        table=pd.DataFrame({'thresholds':{f'{100*a:g}%':v for a,v in values.items()}})
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.csv.tmp')
        table.to_csv(temporary)
        temporary.replace(path)
        low=getattr(self,'alpha_low','5%'); high=getattr(self,'alpha_high','10%')
        # Preserve legacy names and units while accommodating other requested alphas.
        self.threshold=float(table.loc[low,'thresholds']) if low in table.index else values[min(values)]
        self.threshold_high=float(table.loc[high,'thresholds']) if high in table.index else values[max(values)]
        return table


class GWASThresholds(EffectiveTestsMixin):
    """Threshold adapter for arrays, PLINK prefixes, Zarr or genotype sources."""
    def __init__(self, genotypes, *, path='.', alpha_low='5%', alpha_high='10%',
                 legacy_threshold_estimator=None):
        """Initialize GWASThresholds with the inputs and resource settings below.

        Parameters
        ----------
        genotypes : array-like or pandas.DataFrame
            Genotype values with animals in rows and variants in columns, or the genotype source
            accepted by this entry point. Biallelic dosage uses 0/1/2; preserve sample/variant
            labels and the coding expected by the selected kernel or association model.

        path : str or os.PathLike
            Filesystem location read or written by this routine. Use a path to the documented
            file/store format; destination parents and overwrite behavior follow the writer's
            contract. Default: ``'.'``.

        alpha_low : float or array-like
            Numerical scale/retention/step threshold controlling the transformation described above.
            Match vector scales to the affected axis; the exact formula and local comparisons below
            determine its admissible domain. Default: ``'5%'``.

        alpha_high : float or array-like
            Numerical scale/retention/step threshold controlling the transformation described above.
            Match vector scales to the affected axis; the exact formula and local comparisons below
            determine its admissible domain. Default: ``'10%'``.

        legacy_threshold_estimator : callable or None
            Compatibility threshold-estimator hook supplied to GWASThresholds. This constructor
            stores/accepts legacy configuration; see its method dispatch before assuming the
            callable is executed. Default: ``None``. None delegates to the local fallback shown in
            this body; it is not an additional numeric/category value.
        """
        self.genotypes_subset=genotypes
        self.path=Path(path)
        self.alpha_low=alpha_low
        self.alpha_high=alpha_high
        self.legacy_threshold_estimator=legacy_threshold_estimator
