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
    """Build a mapper index from an index_config.yaml entry.

    alevin-fry (via simpleaf) is the implemented path. The kallisto|bustools and
    STARsolo builders are placeholders: those mappers are not implemented end to
    end and are blocked by the mapper gate in mapping.main.validate_samples.
    """
    def __init__(self, gtf_path : str, genome_path: str, out_path: str, work_dir: str=None):
        """`gtf_path`/`genome_path` may be lists, merged into one reference.

        `work_dir` is the scratch root for the build, falling back to $TMPDIR (set
        per-job by SLURM) then the system temp dir. piscem index builds are
        unreliable on networked filesystems, so the build runs there and is
        copied to `out_path` afterwards.
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
        """Whether two paths sit on the same filesystem, so a rename would be atomic."""
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
        """Resolve this index's GTF/FASTA, merging multiple references; returns (gtf, fasta)."""
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
        """STAR's recommended --genomeSAindexNbases for this genome length, clamped to 4..14."""
        genome_size = sum(len(record.seq) for record in SeqIO.parse(self.genome_path, "fasta"))
        genomeSAindex_size = max(int(min(14, (math.log2(genome_size)/2) - 1)), 4)
        return genomeSAindex_size

    @log_exceptions()
    def build_alevin_index(self, threads: int = 16, extra_args=None) -> None:
        """
        Build an index for alevin-fry via `simpleaf index`.

        `--ref-type` and `--rlen` come from the index entry's `extra_index_args`
        rather than being hardcoded here; simpleaf rejects a flag given twice.

        Targets simpleaf >= 0.16, where piscem is the default and `--use-piscem`
        no longer exists (pass `--no-piscem` via extra_index_args to opt out).

        `--work-dir` is set explicitly under the scratch root unless the caller
        supplied it: simpleaf and `piscem build` default it to the relative
        `./workdir.noindex`, i.e. the cwd -- the repo root on networked storage
        when launched from batch_scripts/run_mapper.sh, which the scratch staging
        exists to avoid.
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
        Build a kallisto|bustools index using `kb ref`. Placeholder: kb is not
        implemented end to end.

        Assumes `kb ref` writes the transcript-to-gene map (`t2g_path`) and cDNA
        FASTA (`f1_path`) itself.
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
        Build a STARsolo index with STAR's genomeGenerate mode, sizing
        --genomeSAindexNbases from the genome. Placeholder: STARsolo is not
        implemented end to end.
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
        rather than a truncated index. `force` rebuilds even if the path exists.
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

