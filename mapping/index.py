#!/usr/bin/env python3

import os
import shutil
import math
import logging
import tempfile

from pathlib import Path
from Bio import SeqIO
from shared.utils import run_cmd, log_exceptions


class IndexBuilder:
    """
    IndexBuilder
    ------------

    Interface for building indices for single-cell RNA-seq mappers:
    - alevin-fry (via simpleaf)
    - kallisto|bustools
    - STARsolo

    Each method runs the appropriate external tool with reasonable defaults and optional overrides.
    Configurations are extracted from index_config.yaml.
    """
    def __init__(self, gtf_path : str, genome_path: str, out_path: str, work_dir: str=None):
        """
        Initialize the IndexBuilder.

        Parameters:
        - gtf_path (str or Path): Path to the input GTF annotation file.
        - genome_path (str or Path): Path to the reference genome FASTA file.
        - out_path (str or Path): Directory to write output index files.
        - work_dir (str or Path, optional): Scratch root for index building. Falls
          back to $TMPDIR (set per-job by SLURM) then the system temp dir. piscem
          index builds are unreliable on networked filesystems, so the build runs
          here and is copied to out_path afterwards.
        """

        self.logger = logging.getLogger('SingleCellMappingPipeline')
        self.gtf_path = gtf_path
        self.genome_path = genome_path

        self.out_path = Path(out_path)
        self.work_root = Path(work_dir or os.environ.get('TMPDIR') or tempfile.gettempdir())
        self.gtf_path, self.genome_path = self._prepare_reference_files()


    def _nearest_existing(self, path: Path) -> Path:
        """Walk up until a path that exists, so st_dev can be read."""
        path = Path(path).absolute()
        for candidate in [path, *path.parents]:
            if candidate.exists():
                return candidate
        return Path(path.anchor or '.')

    def _same_filesystem(self, a: Path, b: Path) -> bool:
        """Whether two paths sit on the same filesystem, so a rename would be atomic.

        Args:
            a (Path): First path.
            b (Path): Second path.

        Returns:
            bool: True when they share a device.
        """
        try:
            return os.stat(str(self._nearest_existing(a))).st_dev == os.stat(str(self._nearest_existing(b))).st_dev
        except OSError:
            return False

    def _staging_dir(self, final: Path):
        """
        Where to build the index, or None to build directly at `final`.

        Staging only helps when the scratch root is on a different filesystem
        than the target; if they match, copying a multi-GB index buys nothing.
        """
        if self._same_filesystem(self.work_root, final):
            self.logger.info(
                '[INDEX] Scratch root "{}" is on the same filesystem as "{}"; building in place'.format(
                    self.work_root, final))
            return None
        return self.work_root / '{}_idx_build_{}'.format(final.name, os.getpid())

    def _promote(self, staged: Path, final: Path) -> None:
        """
        Copy a completed index from scratch to its final location.

        Copies to a `.partial-<pid>` sibling first, then renames within the target
        filesystem, which is atomic. A crashed or killed job therefore leaves either
        no index or a complete one -- never a truncated directory at `final`, which
        build_if_missing would mistake for a finished index and skip forever.
        """
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.parent / '{}.partial-{}'.format(final.name, os.getpid())
        if partial.exists():
            shutil.rmtree(str(partial), ignore_errors=True)

        self.logger.info('[INDEX] Copying staged index from "{}" to "{}"'.format(staged, final))
        try:
            shutil.copytree(str(staged), str(partial))
            if final.exists():
                # Forced rebuild: move the old index aside so the rename can land,
                # and keep it until the new one is in place.
                old = final.parent / '{}.old-{}'.format(final.name, os.getpid())
                os.replace(str(final), str(old))
                try:
                    os.replace(str(partial), str(final))
                except Exception:
                    os.replace(str(old), str(final))
                    raise
                shutil.rmtree(str(old), ignore_errors=True)
            else:
                os.replace(str(partial), str(final))
        finally:
            if partial.exists():
                shutil.rmtree(str(partial), ignore_errors=True)


    @log_exceptions()
    def _prepare_reference_files(self):
        """Resolve this index's FASTA/GTF, merging multiple references when asked.

        Returns:
            tuple: (fasta (Path), gtf (Path)).
        """
        gtf_paths = [self.gtf_path] if isinstance(self.gtf_path, str) else self.gtf_path
        genome_paths = [self.genome_path] if isinstance(self.genome_path, str) else self.genome_path

        if len(gtf_paths) == 1 and len(genome_paths) == 1:
            return gtf_paths[0], genome_paths[0]
        elif len(gtf_paths) != len(genome_paths):
            self.logger.error('[INDEX] Number of genomes "{}" does not match number of gtf "{}"'.format(len(genome_paths), len(gtf_paths)))
            raise ValueError('Number of genomes must match number of GTF')
        else:
            # Built under the scratch root, not inside out_path: creating anything
            # inside out_path would make build_if_missing's existence check true and
            # skip the build. Scratch also keeps the merged multi-GB FASTA off the
            # networked filesystem, and lets the job's TMPDIR teardown clean it up.
            temp_dir = self.work_root / '{}_merge_{}'.format(self.out_path.name, os.getpid())
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info('[INDEX] Merging {} references under "{}"'.format(len(genome_paths), temp_dir))

            # Set merged paths
            merged_gtf = temp_dir / '_merged.gtf'
            merged_fasta = temp_dir / '_merged.fasta'

        with open(merged_gtf, 'w') as out_gtf:
            for gtf in gtf_paths:
                with open(gtf) as f:
                    out_gtf.write(f.read())

        with open(merged_fasta, 'w') as out_fasta:
            for fasta in genome_paths:
                with open(fasta) as f:
                    out_fasta.write(f.read())
        return merged_gtf, merged_fasta


    @log_exceptions()
    def _compute_genomeSAindexNbases(self) -> int:
        """
        Compute STAR's recommended value for --genomeSAindexNbases
        based on the total length of the reference genome.

        Returns:
        - int: Computed value for --genomeSAindexNbases (minimum 4, max 14)
        """
        genome_size = sum(len(record.seq) for record in SeqIO.parse(self.genome_path, "fasta"))
        genomeSAindex_size = max(int(min(14, (math.log2(genome_size)/2) - 1)), 4)
        return genomeSAindex_size

    @log_exceptions()
    def build_alevin_index(self, threads: int = 16, extra_args=None) -> None:
        """
        Build an index for alevin-fry via `simpleaf`.

        This method runs `simpleaf index` with common defaults

        Parameters:
        - threads (int): Number of threads to use (default: 16)
        - extra_args (list of str, optional): Additional arguments to pass to simpleaf index

        `--ref-type` and `--rlen` come from the index entry's `extra_index_args`
        rather than being hardcoded here; simpleaf rejects a flag given twice.

        Targets simpleaf >= 0.16, where piscem is the default and `--use-piscem`
        no longer exists (pass `--no-piscem` via extra_index_args to opt out).

        `--work-dir` is set explicitly under the scratch root. simpleaf passes it
        through to `piscem build`, and both default it to the *relative* path
        `./workdir.noindex` -- so leaving it unset drops piscem's temporary build
        files into whatever the process cwd happens to be. batch_scripts/run_mapper.sh cds to the
        repo root, so that is the repo root on Lustre: the exact networked-filesystem
        placement the scratch staging in build_if_missing exists to avoid, and a
        stray `workdir.noindex/` left behind at the top of the tree after every
        build. Skipped if the caller already supplied the flag, since simpleaf
        rejects a flag given twice.
        """
        extra_args = [str(a) for a in extra_args] if extra_args else []
        cmd = ['simpleaf', 'index', '--output', str(self.out_path), '--fasta', str(self.genome_path), '--gtf', str(self.gtf_path), '--threads', str(threads)]

        work_dir = None
        if not any(a in ('-w', '--work-dir') or a.startswith('--work-dir=') for a in extra_args):
            self.work_root.mkdir(parents=True, exist_ok=True)
            work_dir = Path(tempfile.mkdtemp(prefix='simpleaf_workdir_', dir=str(self.work_root)))
            cmd += ['--work-dir', str(work_dir)]

        cmd += extra_args
        self.logger.info('[INDEX] Running: {}'.format(' '.join(cmd)))
        try:
            run_cmd(cmd)
        finally:
            # Cleaned on failure too, matching build_if_missing's handling of the
            # staging dir: these are multi-GB intermediates with no diagnostic value.
            if work_dir is not None:
                shutil.rmtree(str(work_dir), ignore_errors=True)

    @log_exceptions()
    def build_kallisto_index(self, t2g_path: str = 't2g.txt', f1_path: str = 'cdna.fasta', extra_args=None) -> None:
        """
        Build a kallisto|bustools index using `kb ref`.

        This method assumes that `kb ref` will generate the transcript-to-gene map
        (t2g.txt) and cDNA FASTA (cdna.fasta) automatically if paths are provided.

        Parameters:
        - t2g_path (str): Output path for the generated transcript-to-gene map file (default: 't2g.txt')
        - f1_path (str): Output path for the generated cDNA FASTA file (default: 'cdna.fasta')
        - extra_args (list of str, optional): Additional arguments to pass to kb (e.g., ['--workflow', 'nac'])
        """
        index_path = self.out_path / "kb_index.idx"
        cmd = ['kb', 'ref', str(self.genome_path), str(self.gtf_path), '-i', str(index_path), '-g', str(t2g_path), '-f1', str(f1_path)]
        if extra_args:
            cmd += extra_args
        self.logger.info('[INDEX] Running: {}'.format(' '.join(cmd)))
        run_cmd(cmd)

    @log_exceptions()
    def build_star_index(self, extra_args=None) -> None:
        """
        Build a STARsolo index by running STAR's genomeGenerate mode.

        Computes --genomeSAindexNbases based on input genome size.

        Parameters:
        - extra_args (list of str, optional): Additional arguments to pass to STAR
        """
        sa_index_size = self._compute_genomeSAindexNbases()
        cmd = ['STAR', '--runMode', 'genomeGenerate', '--genomeDir', str(self.out_path), '--genomeFastaFiles', str(self.genome_path), '--sjdbGTFfile', str(self.gtf_path), '--runThreadN', '16', '--genomeSAindexNbases', str(sa_index_size)]
        if extra_args:
            cmd += extra_args
        self.logger.info('[INDEX] Running: {}'.format(' '.join(cmd)))
        run_cmd(cmd)

    @log_exceptions()
    def build_if_missing(self, mapper: str, extra_args=None, force=False):
        """Build the index unless one already exists at the target path.

        Treats any existing path as a finished index, which is why the build stages
        through scratch and renames atomically -- a killed build must leave nothing
        rather than a truncated index.

        Args:
            mapper (str): Which mapper's index to build.
            extra_args (list[str], optional): Passed through to the builder.
            force (bool): Rebuild even if the path exists.

        Returns:
            Path: The index directory.
        """

        if self.out_path.exists() and not force:
            self.logger.info('[INDEX] Index already exists at {}'.format(self.out_path))
            return

        # YAML leaves numeric args as ints (e.g. --sjdbOverhang 99); subprocess
        # requires every argv element to be a string.
        if extra_args:
            extra_args = [str(a) for a in extra_args]

        builders = {
            'alevin': self.build_alevin_index,
            'kb': self.build_kallisto_index,
            'starsolo': self.build_star_index}
        if mapper not in builders:
            raise ValueError('[INDEX] Unknown mapper type: {}'.format(mapper))

        final = self.out_path
        staged = self._staging_dir(final)

        # The builders read self.out_path, so point it at scratch for the duration
        # and restore it afterwards regardless of outcome.
        build_target = staged if staged is not None else final
        build_target.mkdir(parents=True, exist_ok=True)
        self.out_path = build_target
        try:
            if staged is not None:
                self.logger.info('[INDEX] Building index in scratch "{}"'.format(staged))
            builders[mapper](extra_args=extra_args)
            if staged is not None:
                self._promote(staged, final)
        finally:
            self.out_path = final
            if staged is not None:
                shutil.rmtree(str(staged), ignore_errors=True)

