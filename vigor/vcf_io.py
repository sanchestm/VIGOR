"""Incremental VCF/BCF to canonical genotype conversion."""
from pathlib import Path
import numpy as np
import pandas as pd

def iter_vcf_dosage(path, *, block_size='auto', maxmem=None, memory_fraction=None):
    """Iterate over vcf dosage.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    block_size : int or 'auto'
        Number of variants or rows processed together by the local block iterator. Supply a
        positive count; smaller blocks reduce peak memory at the cost of more iteration
        overhead. See the called iterator for which axis is blocked. Default: ``'auto'`` (sized
        from the memory budget; integers are ceilings).

    maxmem : int, float, unit string or None
        Working-memory budget; see ``resources.working_memory_bytes``. Default: ``None``.

    memory_fraction : float or None
        Fraction of physical RAM used when maxmem is None; None means 0.80. Default: ``None``.

    Yields
    ------
    object
        Successive blocks or values in iterator order, following the representation described above.
    """
    try: import pysam
    except ImportError as exc: raise ImportError("pysam is required for VCF/BCF conversion") from exc
    vf = pysam.VariantFile(str(path)); samples = np.asarray(list(vf.header.samples), dtype=str)
    from .resources import resolve_variant_chunk
    # boxed Python float + list slot (32) and the float32 output (4) per call; metadata tuple per variant
    block_size = resolve_variant_chunk(block_size, len(samples), bytes_per_call=36, bytes_per_variant=512,
                                       maxmem=maxmem, memory_fraction=memory_fraction)
    rows=[]; cols=[]
    for rec in vf.fetch():
        if len(rec.alts or ()) != 1: continue
        g=[]
        for s in samples:
            gt = rec.samples[str(s)].get("GT")
            g.append(np.nan if gt is None or any(a is None for a in gt) else float(sum(gt)))
        rows.append((str(rec.contig), int(rec.pos), str(rec.ref), str(rec.alts[0]), str(rec.id or f"{rec.contig}:{rec.pos}:{rec.ref}:{rec.alts[0]}")))
        cols.append(g)
        if len(cols) >= block_size:
            yield samples, pd.DataFrame(rows, columns=["chromosome","position","ref","alt","variant_id"]), np.asarray(cols, np.float32).T
            rows=[]; cols=[]
    if cols: yield samples, pd.DataFrame(rows, columns=["chromosome","position","ref","alt","variant_id"]), np.asarray(cols, np.float32).T

__all__=["iter_vcf_dosage"]
