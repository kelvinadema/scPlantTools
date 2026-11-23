import itertools

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from plotnine import *


class AnnotationPlotter:
    """Figures summarising one dataset's annotation result."""
    def __init__(self, annotation_df, ontology, out_dir, logger):
        self.annotation = annotation_df
        self.ontology = ontology
        self.out_dir = out_dir
        self.logger = logger

    
    def _plot_species_support(self):
        """Boxplot of how many reference species backed each called cell type; writes support.png."""
        p = ggplot(self.annotation, aes(x="po_name", y="support_species")) + coord_flip() + geom_boxplot(fill="grey", alpha=0.6, color = 'black') + theme_bw() + labs(title="Number of supporting species per cell type", x="Cell type", y="# Supporting species")
        p.save(filename= '{}/support.png'.format(self.out_dir), dpi=600, width=6, height=4)
        return p

    def _plot_coenrichment(self, cluster_col="cluster", celltype_col="po_name", min_count=2, normalize=True, figsize=(8, 8), prop_threshold=0.05):
        """Network of cell types called together in the same cluster; writes coenrichment.png.

        Edge weight is the number of clusters a pair co-occurs in, as a fraction
        of all clusters when `normalize` (then `prop_threshold` applies, else
        `min_count`). Returns the figure, or None when no edge survives.
        """
        co_counts = {}
        clusters_seen = self.annotation[cluster_col].nunique()
        for cluster, subdf in self.annotation.groupby(cluster_col):
            terms = subdf[celltype_col].dropna().unique().tolist()
            if len(terms) > 1:
                for a, b in itertools.combinations(sorted(terms), 2):
                    co_counts[(a, b)] = co_counts.get((a, b), 0) + 1
        if not co_counts:
            self.logger.warning("[ANNOTATION] No co-enrichments found; skipping network plot")
            return None
        co_df = pd.DataFrame([{"celltype1": a, "celltype2": b, "count": c} for (a, b), c in co_counts.items()])

        if normalize:
            co_df["count"] = co_df["count"] / clusters_seen
            # min_count is a raw-count cut-off and is meaningless once counts
            # are proportions; prop_threshold is its normalized counterpart.
            if prop_threshold is not None:
                co_df = co_df[co_df["count"] >= prop_threshold]
        else:
            co_df = co_df[co_df["count"] >= min_count]

        if co_df.empty:
            self.logger.warning("[ANNOTATION] No co-enrichment edges survived filtering; "
                                "skipping network plot")
            return None

        G = nx.Graph()
        for _, row in co_df.iterrows():
            G.add_edge(row.celltype1, row.celltype2, weight=row["count"])
        pos = nx.circular_layout(G)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(weights)
        widths = [2 + (w / max_w) * 8 for w in weights]

        nx.draw_networkx_edges(G, pos, width=widths, edge_color=weights, edge_cmap=plt.cm.viridis, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="lightblue", edgecolors="black")
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized frequency" if normalize else "Raw count", fontsize=10)

        plt.title("Cell type co-enrichment network", fontsize=12)
        plt.tight_layout()
        fig.savefig('{}/coenrichment.png'.format(self.out_dir), dpi=600, bbox_inches="tight")
        return fig

    def plotAnnotation(self):
        """Draw and write every annotation figure."""
        # Figures are a by-product; the annotation TSV is already on disk by
        # the time this runs, so a plotting fault must not fail the sample.
        for name, fn in (("species support", self._plot_species_support),
                         ("co-enrichment", self._plot_coenrichment)):
            try:
                fn()
            except Exception as exc:
                self.logger.warning("[ANNOTATION] %s plot failed: %s", name, exc)
