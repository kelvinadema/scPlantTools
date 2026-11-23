#!/usr/bin/env python3

import subprocess
import yaml
import logging
import functools
import datetime
import os
import json

from pathlib import Path


# The logger name the current process's entrypoint configured. `setup_logger`
# sets this; `active_logger()` reads it.
#
# `setup_logger` sets `propagate = False`, so a message sent to any other named
# logger has no handler and is lost. Five names are in play
# (SingleCellMappingPipeline, AnnotationLogging, IntegrateLogging, MarkerLogging,
# and refine's reuse of AnnotationLogging), so a shared/ helper cannot hardcode
# one; it asks instead.
_ACTIVE_LOGGER_NAME = 'SingleCellMappingPipeline'


def active_logger() -> logging.Logger:
    """
    The logger this process's entrypoint configured via `setup_logger`.

    Use this in `shared/` helpers instead of naming a pipeline's logger, so a
    helper called from annotation reaches annotation's handlers rather than
    mapping's. Before any `setup_logger` call it resolves to the mapping name.
    """
    return logging.getLogger(_ACTIVE_LOGGER_NAME)


def run_cmd(cmd: list, logfile: str=None, cwd: str=None) -> None:
    """
    Run a subprocess, logging its captured stdout/stderr on failure, then re-raise.

    `logfile` names a logger explicitly; left as None it uses whichever logger
    the entrypoint configured. Do not pass 'SingleCellMappingPipeline' by hand
    from a non-mapping stage: its messages would be lost.
    """
    logger = logging.getLogger(logfile) if logfile else active_logger()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}\nSTDOUT:\n{e.stdout.decode()}\nSTDERR:\n{e.stderr.decode()}")
        raise


def load_config(config_path: str) -> dict:
    """
    Load a general YAML file and return its full contents as a dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)

def merged_obo_bytes(po_path, extension_paths=()) -> bytes:
    """
    The Plant Ontology text with extension term stanzas appended, as bytes ready
    for `pronto.Ontology(io.BytesIO(...))`.

    Pure text concatenation, deliberately: it imports no `pronto` and lives here
    in `shared/` so both ends of the pipeline can call it -- annotation (which
    has pronto) and refinement (which may not, and wraps its own pronto call in
    a try/except). Merging as text rather than through pronto's own API means an
    extension term's `is_a: PO:...` / `relationship: part_of PO:...` lines are
    resolved by pronto's parser against the PO terms in the same document, so a
    grafted term is indistinguishable from a native one downstream. See
    `annotation/nodule.obo` for the worked example.

    An extension file's own OBO header (everything before its first `[Term]`) is
    stripped: a second `format-version:`/`ontology:` header mid-document is
    invalid, and the extension keeps a header only so it validates and edits as
    a standalone file.
    """
    text = Path(po_path).read_text()
    for ext in extension_paths or ():
        raw = Path(ext).read_text()
        # First line that *starts* a stanza ('[Term]', '[Typedef]'). Matching a
        # bare '[Term]' anywhere is wrong: the extension's header comments may
        # mention it in prose ('one [Term] per cell type'), and slicing there
        # would feed that comment to the OBO parser as a broken stanza.
        lines = raw.splitlines(keepends=True)
        start = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith('[')), None)
        stanzas = ''.join(lines[start:]) if start is not None else raw
        text = text.rstrip() + '\n\n' + stanzas
    return text.encode('utf-8')


def load_json_file(path):
    """Read a JSON file; {} (after logging) when it is missing or unparseable."""
    logger = active_logger()
    if not path.exists():
        logger.warning(f"[MAPPING_STATS] file not not found at {path}")
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse json at {path}: {e}")
        return {}

def resolve_run_id() -> str:
    """
    Stable id for one run: $SLURM_JOB_ID under SLURM, else a %Y%m%d-%H%M%S
    timestamp for a local/interactive invocation. Matches each
    batch_scripts/run_*.sh's own logs/<pipeline>-<jobid>.out/.err naming, so a
    run's SLURM output and its own structured log can be cross-referenced by
    run id.
    """
    return os.environ.get('SLURM_JOB_ID') or datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def pipeline_log_path(pipeline: str, log_dir='logs') -> str:
    """
    Standard per-run log path: {log_dir}/{pipeline}/{pipeline}-{run_id}.log.

    The one place this naming convention lives, so the stages cannot drift
    apart. The run id is what gives each run its own file: `setup_logger`'s
    `FileHandler` appends, so a fixed name would interleave unrelated runs.

    Creates the per-pipeline subdirectory. `log_dir` itself is the caller's
    concern -- each entrypoint resolves it from its own config (`log_dir:` in
    pipeline_config.yaml / annotation_config.yaml / refine_config.yaml),
    defaulting to the repo-root `logs/`.
    """
    path = Path(log_dir) / pipeline / '{}-{}.log'.format(pipeline, resolve_run_id())
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def setup_logger(name = 'SingleCellMappingPipeline', log_file=None, level = logging.INFO):
    """
    Configure the pipeline's named logger. Call this exactly once, at the
    entrypoint of the program. Everywhere else, just use
    `logging.getLogger(<name>)` to fetch the already-configured logger rather
    than calling this again -- or, from `shared/`, `active_logger()`.

    Records `name` as the process's active logger so the helpers in this module
    reach the handlers this call attaches. Without that, a `shared/` helper
    invoked from a non-mapping stage logs into a handler-less logger that also
    does not propagate (see `propagate = False` below), and the message is lost.
    """
    global _ACTIVE_LOGGER_NAME
    _ACTIVE_LOGGER_NAME = name

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear() # Prevent duplicate handlers if this is ever called more than once

    # Don't propagate to the root logger - avoids duplicate lines if root
    # ever gets a handler attached (e.g. by a dependency calling
    # logging.basicConfig(), or by pytest/jupyter in interactive use).
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)

    # Optional write to file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    return logger


def log_exceptions(level=logging.ERROR):
    """
    Decorator to log exceptions in any function using the global logger.
    """
    def decorator(func):
        """Wrap `func` so an exception is logged before it propagates."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Call the wrapped function, logging any exception."""
            logger = active_logger()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(level, f"Exception in '{func.__name__}': {e}")
                raise
        return wrapper
    return decorator

def find_root_dir(marker: str=".root"):
    """
    The repo root: `$SCRNASEQ_ROOT` if set, else the nearest ancestor of the cwd
    containing `marker`.

    `$SCRNASEQ_ROOT` takes precedence so this agrees with
    `batch_scripts/lib/common.sh`, which resolves `SC_REPO_ROOT` the same way.
    Without it the two layers disagree: a job submitted with SCRNASEQ_ROOT set
    cds to one tree in bash and resolves a different one in Python.

    Not resolved from `__file__`. This repo is a symlink farm -- jobs run from
    the nobackup tier where `data/`, `results/` and `indices/` live, while the
    code directories are symlinks into the backup tier -- so a `__file__`-based
    walk lands on the physical backup path, where none of the data exists.
    """
    env = os.environ.get('SCRNASEQ_ROOT')
    if env:
        return Path(env)
    path = Path.cwd()
    for parent in [path, *path.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        "Project root marker {!r} not found in {} or any parent -- run from the "
        "repo root, or set $SCRNASEQ_ROOT.".format(marker, Path.cwd()))


def get_sample_output_paths(sample_output_dir: Path, stage: str='mapping', mapper: str=None, create: bool=True) -> dict:
    """
    The standard output paths for one sample, created unless `create=False`.

    `stage='mapping'` (requires `mapper`) gives base/raw/processed/qc/done under
    `{sample_output_dir}/{mapper}/`; `stage='integration'` gives only base.
    """
    if stage == 'mapping':
        if mapper is None:
            raise ValueError("Mapper must be provided for stage='mapping'")
        mapper_dirnames = {'alevin': 'alevin','kb': 'kb','starsolo': 'starsolo'}
        if mapper not in mapper_dirnames:
            raise ValueError('Unknown mapper {}'.format(mapper))
        base = sample_output_dir / mapper_dirnames[mapper]

        paths = {
            'base': base,
            'raw': base / 'raw_matrices',
            'processed': base / 'processed_matrices',
            'qc': base / 'qc',
            'done': base / 'done'}

    elif stage == "integration":
        paths = {
            "base": sample_output_dir}

    else:
        raise ValueError(f"Unknown stage {stage}")

    if create:
        for key, path in paths.items():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
    return paths

def write_done_flag(path):
    """Create a `.done` sentinel recording the completion time; its directory must exist."""
    with open(path, 'w') as f:
        f.write('Completed at {}'.format(datetime.datetime.now().isoformat()))


class GTFparser():
    """Read gene records out of a GTF/GFF3 via a gffutils database."""
    def __init__(self, gtf_path: str, db_path: str=None):
        # Imported here rather than at module scope: gffutils is needed only by
        # this class, but shared/utils.py is imported by every stage for
        # setup_logger/load_config. A module-scope import made the whole module
        # unusable from any environment without a GTF library installed.
        import gffutils

        self.gtf_path = Path(gtf_path)
        self.db_path = db_path or self.gtf_path.with_suffix('.db')

        #Create database if it does not exist
        if not os.path.exists(self.db_path):
            gffutils.create_db(
                str(self.gtf_path),
                dbfn=str(self.db_path),
                force=True,
                keep_order=True,
                merge_strategy='merge',
                sort_attribute_values=True
            )
        self.db = gffutils.FeatureDB(str(self.db_path), keep_order=True)


    def _get_gene_identifier(self, gene):
        """The gene ID for one gffutils feature: gene_id, gene_name or ID, else the feature id."""
        for key in ["gene_id", "gene_name", "ID"]:
            if key in gene.attributes:
                return gene.attributes[key][0]
        return gene.id 
    
    

    def get_genes_by_attribute(self, attr_name: str, attr_value: str, match_type: str = 'exact') -> list:
        """
        IDs of genes whose `attr_name` (e.g. 'gene_biotype', 'product') matches
        `attr_value`; `match_type` is 'exact', 'prefix' or 'substring'.
        """
        result = list()
        for gene in self.db.features_of_type('gene'):
            values = gene.attributes.get(attr_name, [])
            for val in values:
                if (
                (match_type == 'exact' and val == attr_value) or
                (match_type == 'prefix' and val.startswith(attr_value)) or
                (match_type == 'substring' and attr_value in val)
                ):
                    result.append(self._get_gene_identifier(gene))
                    break  # this gene matched; move to the next one
        return result


    def close(self):
        """Close the underlying gffutils database."""
        if self.db is not None:
            self.db.conn.close()
        # db_path is a Path when defaulted in __init__, so compare on .suffix
        # rather than calling str.endswith on it.
        if os.path.exists(self.db_path) and Path(self.db_path).suffix != '.db':
            os.remove(self.db_path)




