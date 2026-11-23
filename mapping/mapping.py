#!/usr/bin/env python3

import logging

from pathlib import Path
from shared.utils import run_cmd, load_config, log_exceptions, write_done_flag, get_sample_output_paths
from shared.samples import resolve_sample, resolve_index_ref
from .mappingutils import BDDataPrepper


class SingleCellMapper:
    """Map one sample's FASTQs with its configured mapper."""
    def __init__(self, sample_name: str, mapper_config_path: str, sample_config_path: str, index_config_path: str, tech_config_path: str, out_path: str):
        self.sample_name = sample_name
        # Copied: _merge_mapper_defaults/_merge_technology_overrides mutate this
        # dict in place, and the resolver hands back its cached entry.
        self.config = dict(resolve_sample(self.sample_name, sample_config_path))
        self.index_dict = load_config(index_config_path)
        self.mapper_config = load_config(mapper_config_path)
        self.tech_config = load_config(tech_config_path)

        self.mapper = self.config['mapper'].lower()
        
        self.paths = get_sample_output_paths(sample_output_dir = Path(out_path).resolve() / self.sample_name, mapper = self.mapper)
        self.output_dir = self.paths['base']

        self.profile = self.config.get('mapping_config', 'default')
        self.tech = self.config.get('tech', '').lower()

        self.logger = logging.getLogger('SingleCellMappingPipeline')

        self._merge_mapper_defaults()
        self._merge_technology_overrides()

        self.index_info = self._resolve_index()    

    @log_exceptions()
    def _merge_mapper_defaults(self) -> None:
        """
        Merge mapper-specific default config (from mapper_config.yaml)
        into the sample config, if not already set.
        """
        defaults = self.mapper_config['mappers'].get(self.mapper, {}).get(self.profile, {})
        for key, val in defaults.items():
            if key not in self.config or self.config[key] is None:
                self.config[key] = val

    @log_exceptions()
    def _merge_technology_overrides(self) -> None:
        """Apply the technology's chemistry/whitelist/geometry over the mapper profile.

        A key present but null counts as unset, so a technology entry cannot blank a
        value the profile supplied.
        """
        tech_entry = self.tech_config.get("technologies", {}).get(self.tech, {})
        mapper_entry = tech_entry.get(self.mapper, {})

        if mapper_entry.get("unsupported", False):
            raise ValueError(f"Mapper '{self.mapper}' is not supported for technology '{self.tech}'")

        for key, val in mapper_entry.items():
            # `is None` matches _merge_mapper_defaults above: a key written with
            # no value counts as unset in both layers.
            if key not in self.config or self.config[key] is None:
                self.config[key] = val

    def _run_alevin(self, min_reads: int = 10, resolution: str = 'cr-like', threads: int = 16):
        """Dispatch to the BD or standard alevin path for this technology."""
        tech = self.config.get('tech', '')
        # BD Rhapsody is the one special case: its cell label is three barcode
        # blocks split by GTGA/GACA linkers, so the per-block whitelists must be
        # merged into one permit list before mapping (_run_alevin_bd). Every other
        # supported chemistry is a single contiguous barcode(+UMI) on R1 against
        # one whitelist -- 10x v2/v3 and DNBelab C4 (two contiguous 10bp blocks =
        # b[20], no linkers) -- so they share the standard path, driven entirely
        # by the technology_config `chemistry` string and `--unfiltered-pl`.
        if tech == 'bd_rhapsody':
            self._run_alevin_bd(min_reads, resolution, threads)
        elif tech in ('10xv3', '10xv2', 'dnbelab_c4'):
            self._run_alevin_standard(min_reads, resolution, threads)
        else:
            raise ValueError(f"Unsupported technology: {tech}")

    @log_exceptions()
    def _resolve_index(self) -> dict:
        """
        Look up and validate the index info based on index_ref.

        Resolution goes through shared.samples.resolve_index_ref so this and
        main.build_required_index cannot disagree about which index a sample
        needs, including a sample-level `index_ref:` override.
        """
        index_ref = resolve_index_ref(self.config, self.mapper_config)
        if not index_ref:
            raise ValueError(
                '[INDEX] Missing "index_ref" for sample: {}. Set it on the sample, or on the '
                'mapper profile mappers.{}.{} in mapping_config.yaml.'.format(
                    self.sample_name, self.mapper, self.profile))
        if index_ref not in self.index_dict['indexes']:
            raise ValueError('[INDEX] Index reference {} not found in index config'.format(index_ref))
        index_entry = self.index_dict['indexes'][index_ref]
        if index_entry.get('mapper') != self.mapper:
            raise ValueError('[INDEX/MAPPING] Index "{}" does not match mapper {}'.format(index_ref, self.mapper))
        return index_entry

    def _run_alevin_bd(self, min_reads: int, resolution: str, threads: int) -> None:        
        """Map BD Rhapsody reads, preparing its split-barcode whitelist first."""
        fq1 = ",".join(self.config["fastq1"])
        fq2 = ",".join(self.config["fastq2"])
        bdprepper = BDDataPrepper(self.config)
        extra_args = bdprepper.PrepBDMapping()
        self._run_simpleaf_quant(fq1=fq1, fq2=fq2, min_reads=min_reads, resolution=resolution,threads=threads, extra_args=extra_args)

    def _run_alevin_standard(self, min_reads: int, resolution: str, threads: int) -> None:
        """Map reads with a single-block whitelist chemistry."""
        # Standard single-whitelist chemistries (10x v2/v3, DNBelab C4): the
        # geometry string and the --unfiltered-pl whitelist come straight from the
        # technology_config entry, with no per-block merging (that is BD-only).
        fq1 = ",".join(self.config["fastq1"])
        fq2 = ",".join(self.config["fastq2"])
        extra_args = list(self.config.get("extra_mapping_args", []))
        self._run_simpleaf_quant(fq1, fq2, min_reads, resolution, threads, extra_args)

    def _run_simpleaf_quant(self, fq1: str, fq2: str, min_reads: int, resolution: str, threads: int, extra_args: list) -> None:
        """Invoke `simpleaf quant` on comma-joined R1/R2 paths; `extra_args` pass through untouched."""
        # Targets simpleaf >= 0.16, where piscem is the default and --use-piscem
        # no longer exists (pass --no-piscem via extra_mapping_args to opt out).
        cmd = [
            'simpleaf', 'quant',
            '--reads1', fq1,
            '--reads2', fq2,
            '--chemistry', self.config.get('chemistry'),
            '--index', self.index_info['idx_path'] + '/index',
            '--output', str(self.output_dir),
            '--min-reads', str(min_reads),
            '--expected-ori', 'fw',
            '--resolution', str(resolution),
            '--threads', str(threads)
        ]
        if extra_args:
            cmd += extra_args
        self.logger.info('[MAPPING] Running: {}'.format(' '.join(cmd)))
        run_cmd(cmd)

    @log_exceptions()
    def _run_kallisto_bustools(self, threads: int=16, workflow: str = 'standard') -> None:
        """Map reads with kb. Gated at the entrypoint; not implemented end to end."""
        fqs = [i for pair in zip(self.config["fastq1"], self.config["fastq2"]) for i in pair]

        cmd = [
            'kb', 'count',
            '-i', self.index_info['idx_path'],
            '-g', self.index_info['t2g_path'],
            '-x', self.config.get('x_flag'),
            '-o', str(self.output_dir),
            '--h5ad',
            '-t', str(threads),
            '--workflow', str(workflow),
            fqs]

        if 'extra_mapping_args' in self.config:
            cmd += self.config["extra_mapping_args"]
        self.logger.info('[MAPPING] Running: {}'.format(' '.join(cmd)))
        run_cmd(cmd)

    @log_exceptions()
    def _run_starsolo(self, threads: int=16) -> None:
        """Map reads with STARsolo. Gated at the entrypoint; not implemented end to end."""
        fq1 = ",".join(self.config["fastq1"])
        fq2 = ",".join(self.config["fastq2"])

        cmd = ['STAR', '--runMode', 'alignReads',
        '--readFilesCommand', 'zcat',
        '--outFileNamePrefix', str(self.output_dir),
        '--genomeDir', self.index_info['idx_path'],
        '--soloType', self.config['solo_type'],
        '--runThreadN', str(threads),
        '--soloUMIdedup', '1MM_CR',
        '--soloUMIfiltering', 'MultiGeneUMI_CR',
        '--soloMultiMappers', 'PropUnique',
        '--clipAdapterType', 'CellRanger4',
        '--soloBarcodeReadLength', '0',
        '--soloCBwhitelist', self.config['soloCBwhitelist'],
        '-readFilesIn', fq2, fq1]
        
        if 'extra_mapping_args' in self.config:
            cmd += self.config["extra_mapping_args"]
        self.logger.info('[MAPPING] Running: {}'.format(' '.join(cmd)))
        run_cmd(cmd)

    @log_exceptions()
    def run(self, force: bool=False) -> None:
        """Map this sample, unless `mapping.done` exists and `force` is False."""
        done_path = self.paths['done'] / 'mapping.done'
        if done_path.exists() and not force:
            self.logger.info('[MAPPING] skipping sample {}; mapping.done exists (use --force-mapping to override)'.format(self.sample_name))
            return

        self.logger.info('[MAPPING] Starting mapping of sample "{}" using mapper "{}"'.format(self.sample_name, self.mapper))
        if self.mapper == 'alevin':
            self._run_alevin()
        elif self.mapper == 'kb':
            self._run_kallisto_bustools()
        elif self.mapper == 'starsolo':
            self._run_starsolo()
        else:
            raise ValueError('[MAPPING] Unknown mapper specified: {}'.format(self.mapper))
        write_done_flag(done_path)
        self.logger.info('[MAPPING] Finished mapping for sample "{}", writing done flag to "{}"'.format(self.sample_name, done_path))