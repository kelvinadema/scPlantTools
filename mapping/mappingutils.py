#!/usr/bin/env python3


import logging
import gzip

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

from plotnine import *
from pathlib import Path
from typing import List
from itertools import product
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle, Patch
from shared.utils import log_exceptions, get_sample_output_paths, load_json_file, write_done_flag
from shared.samples import resolve_sample


def _bezier_ribbon(x0, y0_bot, y0_top, x1, y1_bot, y1_top):
    """
    Closed polygon for one Sankey ribbon: two mirrored cubic-Bezier curves (top
    edge, bottom edge) connecting a vertical span at x0 to a vertical span at
    x1, S-curving through the midpoint. Degenerates to a flat ribbon when the
    two spans are equal, tapers to a point when a span is zero (a merge/split
    node with no width on one side).
    """
    xm = (x0 + x1) / 2
    verts = [(x0, y0_top), (xm, y0_top), (xm, y1_top), (x1, y1_top),
             (x1, y1_bot), (xm, y1_bot), (xm, y0_bot), (x0, y0_bot), (x0, y0_top)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO] + [MplPath.CURVE4] * 3 + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)

class BDDataPrepper():
    """Build BD Rhapsody's combined barcode whitelist from its three block files."""
    def __init__(self, tech_config: str, whitelistdir = None):
        self.logger = logging.getLogger('SingleCellMappingPipeline')
        self.tech_config = tech_config
        if whitelistdir is None:
            # Colocate the merged whitelist with the per-block files it merges
            # (extra_mapping_args' --unfiltered-pl list) rather than hardcoding a
            # path -- those files' own location is already config-driven via
            # technology_config.yaml, so this follows it rather than duplicating it.
            extra_args = list(self.tech_config.get("extra_mapping_args", []))
            if '--unfiltered-pl' in extra_args:
                idx = extra_args.index('--unfiltered-pl')
                first_cls = extra_args[idx + 1].strip().split()[0]
                whitelistdir = Path(first_cls).parent
            else:
                whitelistdir = Path('data/Whitelists')
        self.whitelistdir = Path(whitelistdir)

    @log_exceptions()
    def _merge_cls_files(self, cls_files: List[str], outfn: Path) -> list:
        """Merge the per-block whitelist files into one under `whitelistdir`.

        Returns the merged file's path, reusing an existing one; returns None if
        the block files cannot be parsed.
        """
        outfn = self.whitelistdir / outfn
        if outfn.exists():
            self.logger.info(f"[MAPPING] Merged CLS file exists at {outfn}")
            return outfn
        try:
            if len(cls_files) < 2:
                raise ValueError("Expected at least two CLS files after --unfiltered-pl")
            self._cartesian_product(cls_files, outfn)
            self.logger.info(f"[MAPPING] Merged CLS file written to {outfn}")
            return outfn
        except (ValueError, IndexError) as e:
            self.logger.warning(f"[MAPPING] Failed to parse CLS files for BD rhapsody: {e}")
            return None

    def _cartesian_product(self, files, output_file, delimiter = '') -> str:
        """Write every combination of the blocks' barcodes, one per line."""
        lists = []
        for file in files:
            with open(file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                lists.append(lines)
        
        with open(output_file, 'w') as out:
            for combo in product(*lists):
                out.write(delimiter.join(combo) + '\n')

    def PrepBDMapping(self):
        """`extra_mapping_args` with the `--unfiltered-pl` block files replaced by the merged whitelist."""
        extra_args = list(self.tech_config.get("extra_mapping_args", []))
        idx = extra_args.index('--unfiltered-pl')
        cls_arg = extra_args[idx + 1]
        cls_files = cls_arg.strip().split()
        outfn = self._merge_cls_files(cls_files = cls_files, outfn = 'BD_CLSmerged.txt')
        extra_args[idx + 1] = str(outfn)
        return extra_args


class StatsExtractor():
    """Collect per-sample mapping statistics and render the read-flow figure."""
    def __init__(self, sample_name : str, sample_config_path: str, raw_dir: str):
        self.sample_name = sample_name
        self.sample_cfg = resolve_sample(self.sample_name, sample_config_path)
        self.mapper = self.sample_cfg['mapper'].lower()
        self.fqpaths = self.sample_cfg.get('fastq1', [])
        self.logger = logging.getLogger('SingleCellMappingPipeline')
        
        self.paths = get_sample_output_paths(Path(raw_dir) / self.sample_name, mapper = self.mapper)
        self.out_dir = self.paths['qc']
        self.adata_path = Path(self.paths['processed'], self.sample_name + '_qcfilters.h5ad')

    @log_exceptions()
    def _count_fastq_reads(self) -> int:
        """Total reads across this sample's R1 FASTQs."""
        total = 0
        for path in self.fqpaths:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, 'rt') as f:
                line_count = sum(1 for _ in f)
            total += line_count // 4
        return total

    @log_exceptions()
    def _get_alevin_stats(self):
        """Mapping statistics from alevin-fry's own JSON output."""
        info_path = self.paths['base'] / 'af_map' / 'map_info.json'
        info = load_json_file(info_path)
        return {"total_reads_mapped": info.get("num_mapped"), "total_valid_reads": info.get("num_reads")}

    @log_exceptions()
    def _get_kb_stats(self):
        """Mapping statistics from kb's run info (kb is gated; not implemented end to end)."""
        inspect_path = self.paths['base'] / 'inspect.json'
        inspect = load_json_file(inspect_path)
        runinfo_path = self.paths['base'] / 'run_info.json'
        runinfo = load_json_file(runinfo_path)

        stats = {}
        stats.update({"total_valid_reads": inspect.get("numReads")})
        stats.update({"total_reads_mapped": runinfo.get("n_pseudoaligned"), "total_reads": runinfo.get("n_processed")})
        return stats

    @log_exceptions()
    def _get_starsolo_stats(self):
        """Placeholder for STARsolo's summary statistics; not implemented (STARsolo is gated)."""
        pass

    @log_exceptions()
    def _get_umi_stats(self):
        """Count totals from the processed matrix, per layer when spliced/unspliced exist."""
        stats = {}
        adata = sc.read_h5ad(self.adata_path)
        if all(k in adata.layers for k in ["spliced", "unspliced"]):
            s = adata.layers["spliced"].sum()
            u = adata.layers["unspliced"].sum()
            a = adata.layers["ambiguous"].sum() if "ambiguous" in adata.layers else 0
            c = adata.layers["corrected_counts"].sum() if "corrected_counts" in adata.layers else 0

            stats["spliced"] = int(s)
            stats["unspliced"] = int(u)
            stats["ambiguous"] = int(a)
            stats['corrected'] = int(c)
            stats["total"] = int(s + u + a)
        else:
            x = adata.X.sum()
            stats["total"] = int(x)
        return stats


    def _get_mapping_stats(self) -> dict:
        """Dispatch to this sample's mapper and collect its statistics."""
        if self.mapper == "alevin":
            mapper_stats = self._get_alevin_stats()
        elif self.mapper =='kb':
            mapper_stats = self._get_kb_stats()
        elif self.mapper == 'starsolo':
            mapper_stats = self._get_starsolo_stats()
        else:
            self.logger.error('"{}" is not a valid mapper'.format(self.mapper))
            return {}

        umi_stats = self._get_umi_stats()
        stats = {**umi_stats, **mapper_stats}
        fqsum = self._count_fastq_reads()
        stats['fq_sum'] = fqsum
        return stats

    def _plot_sankey(self, stats, save: bool=True):
        """
        Sankey-style read-flow diagram. No Sankey library is available in this
        env and plotnine has no curve geom, so it is drawn with matplotlib's
        Path/PathPatch Bezier machinery -- a deliberate, scoped exception to the
        plotnine-only plotting convention for this one custom flow diagram.

        Every "loss" branch (Unmapped, Non-informative, Ambient-removed) is an
        exact number from `stats`. The final Spliced/Unspliced/Ambiguous split
        of the Corrected total is a *proportional estimate* (the raw, pre-
        CellBender spliced:unspliced:ambiguous ratio applied to the corrected
        total) -- this pipeline has no real "corrected, split by spliced/
        unspliced" number (those velocity layers are pre-CellBender; `corrected`
        is a separate whole-matrix total), so this is not a fabricated exact
        measurement, just a visual approximation. Labeled as such in the
        report caption, not repeated here to avoid cluttering the figure.
        """
        total_valid = stats['total_valid_reads']
        mapped = stats['total_reads_mapped']
        unmapped = total_valid - mapped
        informative = stats['total']
        mapped_non_informative = mapped - informative
        corrected = stats['corrected']
        # CellBender only removes counts, never adds them; clipped defensively
        # in case rounding/a future correction mode ever pushes this negative.
        ambient_loss = max(0, informative - corrected)
        frac = informative or 1
        spliced, unspliced, ambiguous = (stats.get(k, 0) or 0 for k in ('spliced', 'unspliced', 'ambiguous'))
        corrected_spliced = corrected * spliced / frac
        corrected_unspliced = corrected * unspliced / frac
        corrected_ambiguous = corrected * ambiguous / frac

        color_map = {
            'Total': '#7f7f7f', 'Mapped': '#1f77b4', 'Loss: Unmapped': '#d62728',
            'Loss: Non-informative': '#ff7f0e', 'Loss: Ambient': '#f4c28c', 'Corrected': '#2ca02c',
            'Useful: Spliced': '#9467bd', 'Useful: Unspliced': '#8c564b', 'Ambiguous': '#bcbd22'}

        xs = [0, 1, 2, 3, 4]
        half_w = 0.05

        fig, ax = plt.subplots(figsize=(9, 5))
        # Segments below this fraction of total_valid are too thin for a
        # legible inline label (this pipeline's CellBender ambient removal is
        # often aggressive, so the corrected/spliced/unspliced tail can be a
        # tiny sliver of a many-million-read sample) -- queued into a side
        # callout list instead of drawn inline, so real proportions are kept
        # honest rather than exaggerating small categories to fit a label.
        min_label_frac = 0.03
        small_labels = []

        def node(x, y_bot, y_top, label_key):
            """One Sankey node rectangle."""
            ax.add_patch(Rectangle((x - half_w, y_bot), 2 * half_w, y_top - y_bot,
                                    facecolor=color_map[label_key], edgecolor='none'))

        def ribbon(x0, y0_bot, y0_top, x1, y1_bot, y1_top, label_key):
            """One Sankey ribbon between two nodes."""
            if y0_top <= y0_bot or y1_top <= y1_bot:
                return
            path = _bezier_ribbon(x0 + half_w, y0_bot, y0_top, x1 - half_w, y1_bot, y1_top)
            ax.add_patch(PathPatch(path, facecolor=color_map[label_key], edgecolor='none', alpha=0.65))

        def label(x, y_bot, y_top, value, name):
            """One Sankey node label with its count."""
            pct = 100 * value / total_valid
            if (y_top - y_bot) / total_valid >= min_label_frac:
                ax.text(x, (y_bot + y_top) / 2, '{:,}\n({:.1f}%)'.format(int(value), pct),
                        ha='center', va='center', fontsize=7)
            else:
                small_labels.append('{}: {:,} ({:.1f}%)'.format(name, int(value), pct))

        # Stage 0->1: Total splits into Mapped (bottom, continues) / Unmapped (top, loss).
        node(xs[0], 0, total_valid, 'Total')
        node(xs[1], 0, mapped, 'Mapped')
        node(xs[1], mapped, total_valid, 'Loss: Unmapped')
        ribbon(xs[0], 0, mapped, xs[1], 0, mapped, 'Mapped')
        ribbon(xs[0], mapped, total_valid, xs[1], mapped, total_valid, 'Loss: Unmapped')
        label(xs[0], 0, total_valid, total_valid, 'Total')
        label(xs[1], mapped, total_valid, unmapped, 'Unmapped')

        # Stage 1->2: Mapped splits into Corrected (bottom, continues) /
        # Ambient removed (loss) / Non-informative (loss), bottom-to-top --
        # matching order on both sides of the transition avoids crossing ribbons.
        y_corr0, y_amb0, y_noninf0 = 0, corrected, corrected + ambient_loss
        node(xs[2], 0, corrected, 'Corrected')
        node(xs[2], corrected, corrected + ambient_loss, 'Loss: Ambient')
        node(xs[2], corrected + ambient_loss, mapped, 'Loss: Non-informative')
        ribbon(xs[1], y_corr0, y_amb0, xs[2], 0, corrected, 'Corrected')
        ribbon(xs[1], y_amb0, y_noninf0, xs[2], corrected, corrected + ambient_loss, 'Loss: Ambient')
        ribbon(xs[1], y_noninf0, mapped, xs[2], corrected + ambient_loss, mapped, 'Loss: Non-informative')
        label(xs[2], corrected, corrected + ambient_loss, ambient_loss, 'Ambient removed')
        label(xs[2], corrected + ambient_loss, mapped, mapped_non_informative, 'Non-informative')

        # Stage 2->3: Corrected splits (proportional estimate, see docstring)
        # into Ambiguous / Unspliced / Spliced, bottom-to-top.
        y0 = corrected_ambiguous
        y1 = y0 + corrected_unspliced
        node(xs[3], 0, corrected_ambiguous, 'Ambiguous')
        node(xs[3], y0, y1, 'Useful: Unspliced')
        node(xs[3], y1, corrected, 'Useful: Spliced')
        ribbon(xs[2], 0, corrected_ambiguous, xs[3], 0, corrected_ambiguous, 'Ambiguous')
        ribbon(xs[2], corrected_ambiguous, y1, xs[3], y0, y1, 'Useful: Unspliced')
        ribbon(xs[2], y1, corrected, xs[3], y1, corrected, 'Useful: Spliced')
        label(xs[3], 0, corrected_ambiguous, corrected_ambiguous, 'Ambiguous')
        label(xs[3], y0, y1, corrected_unspliced, 'Unspliced')
        label(xs[3], y1, corrected, corrected_spliced, 'Spliced')

        # Stage 3->4: three colored sub-ribbons flow in (each keeping the same
        # sub-span it had at stage 3 -- this is a re-merge of the same total,
        # not a further split, so source and destination spans match), capped
        # by one solid 'Corrected' node drawn *after* the ribbons so it reads
        # as a clean terminal bar rather than a semi-transparent overlap.
        ribbon(xs[3], 0, corrected_ambiguous, xs[4], 0, corrected_ambiguous, 'Ambiguous')
        ribbon(xs[3], corrected_ambiguous, y1, xs[4], corrected_ambiguous, y1, 'Useful: Unspliced')
        ribbon(xs[3], y1, corrected, xs[4], y1, corrected, 'Useful: Spliced')
        node(xs[4], 0, corrected, 'Corrected')
        label(xs[4], 0, corrected, corrected, 'Corrected total')

        for x, title in zip(xs, ['Total Reads', 'Mapping Status', 'Read Status', 'Read Distribution', 'Final']):
            ax.text(x, -total_valid * 0.04, title, ha='center', va='top', fontsize=9, fontweight='bold')

        if small_labels:
            ax.text(2.3, total_valid * 0.55, 'Corrected breakdown (too thin to label inline):\n' +
                    '\n'.join(small_labels), ha='left', va='top', fontsize=7.5,
                    family='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#cccccc'))

        ax.set_xlim(-0.3, 4.3)
        ax.set_ylim(-total_valid * 0.08, total_valid * 1.02)
        ax.axis('off')
        ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in color_map.items()],
                  loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False, title='Category')
        fig.tight_layout()

        if save:
            fig.savefig('{}/{}_ReadFlow.png'.format(self.out_dir, self.sample_name), dpi=600, bbox_inches='tight')
        plt.close(fig)
        return fig

    def _percentage(self, part, whole, n_digits: int = 2):
        """
        Compute percentage with optional rounding. Returns None if denominator is zero.
        """
        if whole == 0:
            return None
        pct = 100 * float(part) / float(whole)
        return round(pct, n_digits) if n_digits is not None else pct

    def _write_mapping_stats(self, stats):
        """Write MappingStats.tsv."""
        total_valid = stats['total_valid_reads']
        informative = stats.get('total', 0)
        mapped = stats.get('total_reads_mapped', 0)
        corrected = stats.get('corrected', 0)
        unmapped = total_valid - mapped
        mapped_non_informative = mapped - informative
        fq_reads = stats.get('fq_sum')

        entries = [
            ('Spliced', stats.get('spliced', np.nan), True),
            ('Unspliced', stats.get('unspliced', np.nan), True),
            ('Ambiguous', stats.get('ambiguous', np.nan), True),
            ('Mapped (Corrected)', corrected, True),
            ('Mapped (raw)', informative, True),
            ('Mapped (non-info)', mapped_non_informative, True),
            ('Unmapped', unmapped, True),
            ('Total Reads Mapped', mapped, True),
            # Both of these are supersets of "mapped reads" by construction
            # (mapped reads are a subset of valid reads, which are a subset
            # of/equal to raw FASTQ reads) -- expressing them as "% of mapped
            # reads" is guaranteed to read >=100% and isn't a meaningful
            # ratio, so it's left NA rather than computed.
            ('Total Valid Reads', total_valid, False),
            ('Total FASTQ Reads', fq_reads, False)]

        rows = []
        for metric, value, show_pct_mapped in entries:
            if value is None:
                continue
            pct_valid = self._percentage(value, total_valid)
            pct_mapped = self._percentage(value, mapped) if show_pct_mapped else None
            rows.append({
                'Sample': self.sample_name,
                'Metric': metric,
                'Value': value,
                '% of Valid Reads': pct_valid,
                '% of Mapped Reads': pct_mapped
            })

        df = pd.DataFrame(rows)
        df.to_csv(Path(self.out_dir, 'MappingStats.tsv'), sep='\t', index=False)

    def report_stats(self, force: bool=False):
        """Collect the statistics, write the TSV and draw the read-flow figure, unless `mappingstats.done` exists."""
        done_path = self.paths['done'] / 'mappingstats.done'
        if done_path.exists() and not force:
            self.logger.info('[MAPPING] skipping sample "{}"; mappingstats.done exists (use --force-mapextraction to override)'.format(self.sample_name))
            return
        stats = self._get_mapping_stats()
        self._write_mapping_stats(stats)
        self._plot_sankey(stats)
        write_done_flag(done_path)