from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .enrichment import gene_set_enrichment
@dataclass
class ProjectStoryResult:
    """ProjectStoryResult represents project story result.

    Attributes
    ----------
    loci : pd.DataFrame
        Stored loci; its axis/order follows the construction inputs.
    candidate_genes : pd.DataFrame
        Stored candidate genes; its axis/order follows the construction inputs.
    enrichment : pd.DataFrame | None
        Stored enrichment; its axis/order follows the construction inputs.
    metadata : dict
        Stored metadata; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    loci : pd.DataFrame
        Stored loci supplied by the corresponding fitting/planning routine. See that routine for
        the result-axis/value contract. Required construction field.

    candidate_genes : pd.DataFrame
        Stored candidate genes supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    enrichment : pd.DataFrame | None
        Stored enrichment supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    metadata : dict
        Additional result provenance, axis counts and fit/selection configuration. Required
        construction field.
    """
    loci: pd.DataFrame
    candidate_genes: pd.DataFrame
    enrichment: pd.DataFrame|None
    metadata: dict
def integrate_project_story(loci,candidate_genes,*,term_to_genes=None,gene_universe=None,gene_window_bp=1_000_000):
    # Correct default: universe is genes reachable from the tested loci/candidate construction, never whole annotation.
    """Return the integrate project story result computed by ``ProjectStoryResult``.

    Parameters
    ----------
    loci : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source.

    candidate_genes : table, mapping or None
        Optional biological evidence/candidate records integrated by this routine. Match
        locus/gene/trait identifiers and assembly to the primary inputs; None omits that
        evidence source.

    term_to_genes : sequence, mapping or table
        Background gene universe or term/covariate mapping for enrichment. Use the genes
        eligible under the analysis selection process; mismatched universes distort the
        enrichment interpretation. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.

    gene_universe : sequence, mapping or table
        Background gene universe or term/covariate mapping for enrichment. Use the genes
        eligible under the analysis selection process; mismatched universes distort the
        enrichment interpretation. Default: ``None``. None delegates to the local fallback shown
        in this body; it is not an additional numeric/category value.

    gene_window_bp : int or None
        Physical distance/window bound in base pairs for selecting nearby variants/genes.
        Positive bounds restrict the region; None leaves the distance unrestricted where
        supported. Window width versus radius is specified by the summary. Default: ``1000000``.

    Returns
    -------
    object
        Computed result from ``ProjectStoryResult(loci, c, enr, {'gene_window_bp': int(gene_window_bp), 'universe_source': 'tested_reachable_genes'})``; see the summary and the called implementation for its labeled/numerical representation.
    """
    c=candidate_genes.copy();universe=gene_universe
    if universe is None:
        gc=next((x for x in ('gene','gene_id','gene_name') if x in c),None)
        if gc is None:raise ValueError('candidate_genes needs gene/gene_id or explicit gene_universe')
        universe=c[gc].dropna().astype(str).unique().tolist()
    enr=None
    if term_to_genes is not None:
        score_col=next((x for x in ('selected','is_candidate','priority') if x in c),None);gc=next(x for x in ('gene','gene_id','gene_name') if x in c)
        selected=c.loc[c[score_col].astype(bool),gc] if score_col else c[gc]
        enr=gene_set_enrichment(selected,term_to_genes,universe=universe)
    return ProjectStoryResult(loci,c,enr,{'gene_window_bp':int(gene_window_bp),'universe_source':'tested_reachable_genes'})
