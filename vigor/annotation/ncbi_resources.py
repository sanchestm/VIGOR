"""Explicit-accession NCBI FASTA/GTF/GFF resources and chromosome synonyms.

Uses the NCBI datasets CLI, as GWAS-pipeline.pull_NCBI_genome_info does, without
removing existing directories or guessing the assembly from variant positions.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import re
import shutil
import subprocess
import tempfile
import zipfile


@dataclass(frozen=True)
class GenomeResources:
    """GenomeResources represents genome resources.

    Attributes
    ----------
    accession : str
        Stored accession; its axis/order follows the construction inputs.
    fasta_path : Path | None
        Stored fasta path; its axis/order follows the construction inputs.
    gtf_path : Path | None
        Stored gtf path; its axis/order follows the construction inputs.
    gff_path : Path | None
        Stored gff path; its axis/order follows the construction inputs.
    sequence_report_path : Path
        Stored sequence report path; its axis/order follows the construction inputs.
    contig_aliases : dict[str, str]
        Stored contig aliases; its axis/order follows the construction inputs.
    assembly_report_path : Path | None
        Stored assembly report path; its axis/order follows the construction inputs.

    Constructor fields
    ------------------
    accession : str
        Stored accession supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    fasta_path : Path | None
        Stored fasta path supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    gtf_path : Path | None
        Stored gtf path supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    gff_path : Path | None
        Stored gff path supplied by the corresponding fitting/planning routine. See that routine
        for the result-axis/value contract. Required construction field.

    sequence_report_path : Path
        Stored sequence report path supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Required construction field.

    contig_aliases : dict[str, str]
        Stored contig aliases supplied by the corresponding fitting/planning routine. See that
        routine for the result-axis/value contract. Required construction field.

    assembly_report_path : Path | None
        Stored assembly report path supplied by the corresponding fitting/planning routine. See
        that routine for the result-axis/value contract. Default: ``None``.
    """
    accession: str
    fasta_path: Path | None
    gtf_path: Path | None
    gff_path: Path | None
    sequence_report_path: Path
    contig_aliases: dict[str,str]
    assembly_report_path: Path | None = None


def load_ncbi_genome_resources(directory, *, accession):
    """Read one extracted datasets bundle; map chromosome names to FASTA IDs.

    Parameters
    ----------
    directory : path-like, bool or None
        Input/output or cache location for this reader/fitter. Use the artifact format named by
        the routine; None/False disables an optional destination/cache, and True selects a
        default cache directory only where explicitly supported.

    accession : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified.

    Returns
    -------
    object
        Computed result from ``GenomeResources(str(accession), fasta, gtf, gff, report, aliases, assembly_report)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if not re.fullmatch(r'GC[AF]_\d+\.\d+',str(accession)):
        raise ValueError('provide a versioned GCF_/GCA_ assembly accession')
    root=Path(directory)
    candidates=[root/'ncbi_dataset/data'/accession,root/'data'/accession,root/accession,root]
    base=next((p for p in candidates if (p/'sequence_report.jsonl').is_file()),None)
    if base is None:raise FileNotFoundError('NCBI sequence_report.jsonl not found for this accession')
    report=base/'sequence_report.jsonl';aliases={}
    with report.open() as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line)
            canonical=(row.get('genbankAccession') or row.get('refseqAccession')) if str(accession).startswith('GCA_') else (row.get('refseqAccession') or row.get('genbankAccession'))
            canonical=canonical or row.get('sequenceName')
            if not canonical:continue
            names=[row.get('sequenceName'),row.get('refseqAccession'),row.get('genbankAccession')]
            if row.get('role',row.get('sequenceRole','')) in ('','assembled-molecule'):
                names.extend([row.get('chrName'),row.get('assignedMolecule')])
            for name in names:
                if name is None or str(name).lower() in {'un','na','unknown',''}:continue
                name=str(name);normal=name[3:] if name.lower().startswith('chr') else name
                if normal in aliases and aliases[normal]!=canonical:
                    # Unplaced contigs can share an assigned-molecule label;
                    # do not let that label select an arbitrary sequence.
                    raise ValueError(f'ambiguous chromosome synonym: {normal}')
                aliases[normal]=str(canonical)
    def find(patterns):
        """Find one genome resource matching the supplied filename patterns, rejecting ambiguity.

        Parameters
        ----------
        patterns : str, sequence or mapping
            Pattern/filter selectors applied to the resource/table search. Use patterns matching the
            required filenames/header or existing metadata fields.

        Returns
        -------
        object
            Computed result from ``found[0] if found else None``; see the summary and the called implementation for its labeled/numerical representation.
        """
        found=sorted({p for pattern in patterns for p in base.glob(pattern) if p.is_file()})
        if len(found)>1:raise ValueError(f'ambiguous resource paths for {patterns}')
        return found[0] if found else None
    fasta=find(['*_genomic.fna','genomic.fna','*_genomic.fna.gz','genomic.fna.gz'])
    gtf=find(['genomic.gtf','genomic.gtf.gz']);gff=find(['genomic.gff','genomic.gff3','genomic.gff.gz','genomic.gff3.gz'])
    assembly_report=base.parent/'assembly_data_report.jsonl'
    if not assembly_report.is_file():assembly_report=None
    return GenomeResources(str(accession),fasta,gtf,gff,report,aliases,assembly_report)


def download_ncbi_genome_resources(accession, destination, *, redownload=False,
                                    datasets_executable='datasets', timeout=1800):
    """Download matching sequence/annotation resources to a dedicated accession folder.

    Requires the NCBI datasets command-line tool. Cached complete resources are
    reused. The caller supplies the known genotype assembly accession; this
    function never chooses a species/build or deletes unrelated project files.

    Parameters
    ----------
    accession : str or None
        Assembly/species identifier attached to annotation metadata or used to validate
        coordinate compatibility. None leaves the identifier unspecified. Passed to
        ``load_ncbi_genome_resources`` as ``accession``.

    destination : str or os.PathLike
        Filesystem location read or written by this routine. Use a path to the documented
        file/store format; destination parents and overwrite behavior follow the writer's
        contract.

    redownload : bool
        Compatibility flag accepted by this resource downloader. The local implementation
        follows its existing-resource check rather than consulting this flag; see that check
        before assuming forced replacement. Default: ``False``.

    datasets_executable : str
        NCBI datasets executable name/path invoked by the resource downloader. It must resolve
        on PATH or as the requested executable path. Default: ``'datasets'``.

    timeout : float
        Maximum wait in seconds for each network/file request, as passed to the request backend.
        Default: ``1800``. Passed to ``subprocess.run`` as ``timeout``.

    Returns
    -------
    object
        Computed result from ``load_ncbi_genome_resources(target, accession=accession)``; see the summary and the called implementation for its labeled/numerical representation.
    """
    if not re.fullmatch(r'GC[AF]_\d+\.\d+',str(accession)):
        raise ValueError('provide a versioned GCF_/GCA_ assembly accession')
    target=Path(destination)/str(accession)
    if target.exists() and not redownload:
        resources=load_ncbi_genome_resources(target,accession=accession)
        if resources.fasta_path is None or (resources.gtf_path is None and resources.gff_path is None):
            raise ValueError('cached bundle lacks FASTA or gene annotations; request redownload explicitly')
        return resources
    executable=shutil.which(str(datasets_executable))
    if executable is None:raise FileNotFoundError('NCBI datasets CLI not installed; supply datasets_executable or an extracted bundle')
    parent=target.parent;parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ncbi-genome-',dir=parent) as tmp:
        temp=Path(tmp);archive=temp/'genome.zip';extracted=temp/'extracted'
        command=[executable,'download','genome','accession',str(accession),'--include',
                 'genome,gtf,gff3,seq-report','--filename',str(archive)]
        subprocess.run(command,check=True,timeout=timeout,capture_output=True,text=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path=Path(member.filename)
                if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe archive member')
            bundle.extractall(extracted)
        resources=load_ncbi_genome_resources(extracted,accession=accession)
        if resources.fasta_path is None or (resources.gtf_path is None and resources.gff_path is None):
            raise ValueError('downloaded bundle lacks FASTA or gene annotations')
        # Copy validated resources. Existing accession data remain until the
        # replacement is ready; no rm -r or project-wide cleanup is performed.
        shutil.copytree(extracted,target,dirs_exist_ok=True)
        (target/'domino_genome_resources.json').write_text(json.dumps({
            'accession':str(accession),'source':'NCBI datasets','include':'genome,gtf,gff3,seq-report',
            'coordinate_system':'1-based closed'},indent=2))
    return load_ncbi_genome_resources(target,accession=accession)
