#!/usr/bin/env python3
"""
The reviewer's real decision object: a **group of genesets, spanning methods**.

Stage 1 produces 105 genesets across six marker methods on the test target,
which collapse to **18 distinct cell types**. The six methods largely rediscover
the same populations -- `root hair cell` and `root endodermis` are found by all
six -- so a per-geneset decision file asks for 105 answers to about 18
questions, and asks them one method at a time, which is exactly the axis along
which the answer is redundant.

A group is one final cell type: the genesets, from any method, that describe it.
Naming and grouping are **independent**. A group with no `po_id` is a real
population nobody could name, and becomes an `unknown_<group>` class rather
than being absorbed into a neighbour -- scANVI is a closed-world classifier, so
a cell type absent from the label set comes back as the nearest available class,
often confidently (see refine/labels.py).

Three states a geneset can be in, and they are three different statements:

    grouped      part of a named or unnamed cell type
    ungrouped    left alone; becomes its own `unknown_<method>_<geneset>` class
    dropped      the reviewer judged it junk; contributes no seed cells at all

Two interchangeable serialisations, same information, JSON winning when both
are present -- the same convention `shared/decisions.py` established:

  **TSV**, one row per (method, geneset), hand-editable and diffable:

    method    geneset      group           po_id       po_name         note
    wilcoxon  3            root_hair_cell  PO:0000256  root hair cell
    wilcoxon  4            root_hair_cell  PO:0000256  root hair cell
    hotspot   hotspot_M3   root_hair_cell  PO:0000256  root hair cell
    hotspot   hotspot_M13  __UNGROUPED__
    cfmf      cfmf_M7      __DROP__

  **JSON**, what the review page exports.

This module imports no `pronto` and nothing heavy, for the same reason
`shared/decisions.py` does not: the annotation and refinement ends run under
different interpreters, and ontology validation is injected by whichever caller
has the ontology.
"""

import hashlib
import json

from pathlib import Path

import pandas as pd

SCHEMA = 'groups/1'

# `group` column sentinels. Deliberately not empty strings: "I have not decided"
# and "I decided to leave this alone" are different, and only the second should
# survive a round trip through a spreadsheet that helpfully strips blanks.
DROP = '__DROP__'
UNGROUPED = '__UNGROUPED__'

TSV_COLUMNS = ('method', 'geneset', 'group', 'po_id', 'po_name',
               'n_seed_cells', 'decision', 'top_po_name', 'note')

# Excluded from the hash. Editing a comment must not invalidate a refinement
# run; changing a grouping or a call must.
_UNHASHED = ('note', 'po_name', 'n_seed_cells', 'decision', 'top_po_name')


class GroupError(ValueError):
    """Raised for a malformed, inconsistent or stale group document."""


def _slug(name: str) -> str:
    """A PO name to a group id: `root hair cell` -> `root_hair_cell`."""
    out = ''.join(c if c.isalnum() else '_' for c in str(name).strip().lower())
    while '__' in out:
        out = out.replace('__', '_')
    return out.strip('_') or 'group'


class GroupDocument:
    """
    Resolved grouping, plus the provenance needed to detect staleness.

    `groups` is `{group_id: {'po_id', 'po_name', 'note', 'members': [(method, geneset)]}}`.
    `dropped` and `ungrouped` are sets of `(method, geneset)`.
    """

    def __init__(self, sample, groups=None, dropped=None, ungrouped=None,
                 parent_hashes=None, source=None):
        self.sample = sample
        self.groups = groups or {}
        self.dropped = set(dropped or ())
        self.ungrouped = set(ungrouped or ())
        self.parent_hashes = parent_hashes or {}
        self.source = str(source) if source else ''

    # ---------------------------------------------------------------- queries

    @property
    def members(self):
        """{(method, geneset) -> group_id} for every grouped geneset."""
        return {m: gid for gid, g in self.groups.items() for m in g['members']}

    @property
    def named(self):
        """The groups that carry a label, i.e. the ones that seed anything.

        Returns:
            list: The labelled groups.
        """
        return {gid: g for gid, g in self.groups.items() if g.get('po_id')}

    @property
    def unnamed(self):
        """Real populations nobody could name. They still get a class."""
        return {gid: g for gid, g in self.groups.items() if not g.get('po_id')}

    def methods(self):
        """Every marker method referenced by any group.

        Returns:
            list[str]: Method names.
        """
        return sorted({m for g in self.groups.values() for m, _ in g['members']}
                      | {m for m, _ in self.dropped} | {m for m, _ in self.ungrouped})

    @property
    def hash(self):
        """
        Identity of a grouping: membership and names only.

        Sorted throughout so that reordering members in the file -- which a
        spreadsheet or an export will happily do -- is not mistaken for a
        different decision.
        """
        payload = {
            'groups': sorted(
                [gid, g.get('po_id') or '__NONE__',
                 sorted(['{}|{}'.format(*m) for m in g['members']])]
                for gid, g in self.groups.items()),
            'dropped': sorted('{}|{}'.format(*m) for m in self.dropped),
            'ungrouped': sorted('{}|{}'.format(*m) for m in self.ungrouped),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def summary(self) -> pd.DataFrame:
        """A short description of what this document groups.

        Returns:
            str: The summary line.
        """
        rows = [{'group': gid, 'po_id': g.get('po_id', ''), 'po_name': g.get('po_name', ''),
                 'n_members': len(g['members']),
                 'n_methods': len({m for m, _ in g['members']}),
                 'members': ', '.join('{}:{}'.format(*m) for m in sorted(g['members']))}
                for gid, g in sorted(self.groups.items())]
        return pd.DataFrame(rows)

    def __repr__(self):
        """Short representation naming the sample and group count."""
        return ('<GroupDocument {} groups ({} named), {} ungrouped, {} dropped>'.format(
            len(self.groups), len(self.named), len(self.ungrouped), len(self.dropped)))


# ------------------------------------------------------------------- propose

def propose(sample, templates: dict, annotations: dict = None) -> GroupDocument:
    """
    The starting grouping, derived from stage 1's own top-1 calls.

    Genesets whose automatic top call is the same PO term are put in the same
    group. That is a *proposal*, not an answer -- it is exactly the merge the
    old per-method file performed implicitly when two genesets happened to be
    given the same id, made explicit and extended across methods so the
    reviewer can see it and disagree with it.

    A geneset with no call is left **ungrouped** rather than dropped: it is a
    real population that nothing could name, and the two are opposite
    statements. On the test target that is 49 of 105 genesets, and it is where
    the 5,647 cells no method called end up.

    `templates` is `{method: decision template DataFrame}`.
    """
    groups, ungrouped = {}, set()
    for method, template in sorted(templates.items()):
        for _, row in template.iterrows():
            geneset = str(row['cluster'])
            po_id = row.get('top_po_id')
            po_id = '' if po_id is None or pd.isna(po_id) else str(po_id).strip()
            po_name = row.get('top_po_name')
            po_name = '' if po_name is None or pd.isna(po_name) else str(po_name).strip()
            member = (method, geneset)
            if not po_id:
                ungrouped.add(member)
                continue
            gid = _slug(po_name or po_id)
            entry = groups.setdefault(gid, {'po_id': po_id, 'po_name': po_name,
                                            'note': '', 'members': []})
            if entry['po_id'] != po_id:
                # Two different terms sluggifying to the same id. Rare, but a
                # silent collision here would merge two cell types.
                gid = '{}_{}'.format(gid, po_id.replace(':', '').lower())
                entry = groups.setdefault(gid, {'po_id': po_id, 'po_name': po_name,
                                                'note': '', 'members': []})
            entry['members'].append(member)

    parent_hashes = {m: file_hash(p) for m, p in (annotations or {}).items()
                     if Path(p).exists()}
    return GroupDocument(sample, groups=groups, ungrouped=ungrouped,
                         parent_hashes=parent_hashes, source='(proposed)')


def file_hash(path) -> str:
    """Content hash of a file, for staleness checks.

    Args:
        path (Path): The file.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------- write

def to_frame(doc: GroupDocument, evidence: dict = None) -> pd.DataFrame:
    """
    The hand-editable TSV: one row per (method, geneset), every geneset present.

    `evidence` is `{(method, geneset): {...}}` of read-only context -- seed
    count, the automatic call -- carried so the file can be edited without the
    HTML page beside it. None of it is hashed.
    """
    evidence = evidence or {}
    rows = []
    for gid, group in sorted(doc.groups.items()):
        for member in sorted(group['members']):
            rows.append({'method': member[0], 'geneset': member[1], 'group': gid,
                         'po_id': group.get('po_id', ''), 'po_name': group.get('po_name', ''),
                         'note': group.get('note', '')})
    for member in sorted(doc.ungrouped):
        rows.append({'method': member[0], 'geneset': member[1], 'group': UNGROUPED,
                     'po_id': '', 'po_name': '', 'note': ''})
    for member in sorted(doc.dropped):
        rows.append({'method': member[0], 'geneset': member[1], 'group': DROP,
                     'po_id': '', 'po_name': '', 'note': ''})
    frame = pd.DataFrame(rows)
    if len(frame) and evidence:
        # Read-only context so the file can be edited without the HTML page
        # open beside it. Not hashed -- see _UNHASHED.
        for col in ('n_seed_cells', 'decision', 'top_po_name'):
            frame[col] = [evidence.get((r['method'], r['geneset']), {}).get(col, '')
                          for _, r in frame.iterrows()]
    return frame.reindex(columns=[c for c in TSV_COLUMNS if c in frame.columns])


def is_edited(path) -> bool:
    """
    Has a human touched this file?

    Unlike a per-geneset template -- where any non-empty `chosen_po_id` is an
    edit -- a proposed grouping is already fully populated, so "edited" cannot
    be read off emptiness. Compared against the proposal's hash instead, which
    the file records for exactly this purpose.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        if path.suffix == '.json':
            payload = json.loads(path.read_text())
            return payload.get('proposed_hash') not in (None, payload.get('hash'))
        frame = pd.read_csv(path, sep='\t', dtype=str).fillna('')
        return bool(frame.get('note', pd.Series(dtype=str)).str.strip().any())
    except Exception:                                   # noqa: BLE001 - unreadable == precious
        return True


def write(path, doc: GroupDocument, evidence: dict = None, force: bool = False,
          proposed_hash: str = None) -> Path:
    """
    Write the grouping. Refuses to clobber an edited file unless forced.

    Losing a review session to an incidental rerun is the one unrecoverable
    failure in this stage, so the default is to refuse -- same rule as
    `shared.decisions.write_template`.
    """
    path = Path(path)
    if is_edited(path) and not force:
        raise GroupError(
            '{} has been edited and would be overwritten. Move it aside, or pass '
            'force=True to discard those edits.'.format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.json':
        path.write_text(json.dumps(as_payload(doc, proposed_hash=proposed_hash), indent=2))
    else:
        to_frame(doc, evidence).to_csv(path, sep='\t', index=False)
    return path


def as_payload(doc: GroupDocument, proposed_hash: str = None) -> dict:
    """The JSON form, which is also what the review page exports."""
    return {
        'schema': SCHEMA,
        'sample': doc.sample,
        'hash': doc.hash,
        # Lets `is_edited` tell a freshly-proposed file from a reviewed one.
        'proposed_hash': proposed_hash if proposed_hash is not None else doc.hash,
        'parent_hashes': doc.parent_hashes,
        'groups': {gid: {'po_id': g.get('po_id', ''), 'po_name': g.get('po_name', ''),
                         'note': g.get('note', ''),
                         'members': [list(m) for m in sorted(g['members'])]}
                   for gid, g in sorted(doc.groups.items())},
        'dropped': [list(m) for m in sorted(doc.dropped)],
        'ungrouped': [list(m) for m in sorted(doc.ungrouped)],
    }


# ---------------------------------------------------------------------- read

def read(path, available: dict = None, valid_po_ids=None, logger=None) -> GroupDocument:
    """
    Read a grouping (TSV or JSON) and validate it against what stage 1 produced.

    `path` may be a directory, in which case `*_groups.json` wins over
    `*_groups.tsv` -- the JSON is the review page's export, so if a human
    produced one it supersedes the file they started from.

    `available` is `{method: [geneset, ...]}`. Validation, and why each
    severity:

    - a geneset the marker method never produced **raises** -- a typo or a
      stale file.
    - a geneset missing from the document **raises**, naming every one at
      once. Silence here would mean cells quietly contributing no seeds, which
      is indistinguishable from a decision.
    - two rows of one group disagreeing about `po_id` **raises**. It is
      ambiguous by construction and guessing which one is meant would silently
      name a cell type.
    - a `po_id` absent from `valid_po_ids` **raises**, when the caller has an
      ontology to check against.
    - a group whose members all come from one method only **warns**: legitimate
      (only cfmf found this population) but worth seeing.
    """
    path = Path(path)
    if path.is_dir():
        js = sorted(path.glob('*_groups.json'))
        tsv = sorted(path.glob('*_groups.tsv'))
        if not js and not tsv:
            raise GroupError('No *_groups.json or *_groups.tsv in {}'.format(path))
        path = js[0] if js else tsv[0]

    doc = _read_json(path) if path.suffix == '.json' else _read_tsv(path)
    doc.source = str(path)

    if available is not None:
        known = {(m, str(g)) for m, sets in available.items() for g in sets}
        seen = set(doc.members) | doc.dropped | doc.ungrouped
        unknown = sorted(seen - known)
        if unknown:
            raise GroupError(
                '{}: {} geneset(s) are not in the marker tables, e.g. {}. A stale '
                'grouping, or a typo.'.format(path, len(unknown),
                                              ', '.join('{}:{}'.format(*u) for u in unknown[:5])))
        absent = sorted(known - seen)
        if absent:
            raise GroupError(
                '{}: {} geneset(s) have no row, e.g. {}. Every geneset needs one -- use '
                '{} to leave one out deliberately, or {} to give it its own class.'.format(
                    path, len(absent), ', '.join('{}:{}'.format(*a) for a in absent[:5]),
                    DROP, UNGROUPED))

    if valid_po_ids is not None:
        bad = sorted({g['po_id'] for g in doc.groups.values()
                      if g.get('po_id') and g['po_id'] not in valid_po_ids})
        if bad:
            raise GroupError(
                '{}: {} are not Plant Ontology terms in the configured ontology. Check for '
                'a typo.'.format(path, bad))

    if logger is not None:
        for gid, group in sorted(doc.groups.items()):
            methods = {m for m, _ in group['members']}
            if len(methods) == 1:
                logger.warning(
                    '[GROUPS] %s rests on %s alone -- accepted, but only one method found '
                    'this population', gid, next(iter(methods)))
    return doc


def _read_json(path) -> GroupDocument:
    """Read a JSON file.

    Args:
        path (Path): The file.

    Returns:
        dict: The parsed contents.
    """
    payload = json.loads(Path(path).read_text())
    schema = payload.get('schema')
    if schema and schema != SCHEMA:
        raise GroupError('{}: schema {!r}, expected {!r}'.format(path, schema, SCHEMA))
    groups = {gid: {'po_id': (g.get('po_id') or '').strip(),
                    'po_name': g.get('po_name', ''), 'note': g.get('note', ''),
                    'members': [tuple(m) for m in g.get('members', [])]}
              for gid, g in (payload.get('groups') or {}).items()}
    empty = sorted(gid for gid, g in groups.items() if not g['members'])
    if empty:
        # A group with no members labels nothing; keeping it would put an empty
        # class in front of scANVI.
        raise GroupError('{}: group(s) {} have no members'.format(path, empty))
    return GroupDocument(
        payload.get('sample', ''), groups=groups,
        dropped={tuple(m) for m in payload.get('dropped', [])},
        ungrouped={tuple(m) for m in payload.get('ungrouped', [])},
        parent_hashes=payload.get('parent_hashes') or {})


def _read_tsv(path) -> GroupDocument:
    """Read a TSV into a DataFrame.

    Args:
        path (Path): The file.

    Returns:
        pandas.DataFrame: The contents.
    """
    frame = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    missing = [c for c in ('method', 'geneset', 'group') if c not in frame]
    if missing:
        raise GroupError('{}: missing column(s) {}'.format(path, missing))

    groups, dropped, ungrouped = {}, set(), set()
    for _, row in frame.iterrows():
        member = (str(row['method']).strip(), str(row['geneset']).strip())
        gid = str(row['group']).strip()
        if gid == DROP:
            dropped.add(member)
            continue
        if gid in ('', UNGROUPED):
            ungrouped.add(member)
            continue
        po_id = str(row.get('po_id', '')).strip()
        entry = groups.setdefault(gid, {'po_id': po_id, 'po_name': str(row.get('po_name', '')),
                                        'note': str(row.get('note', '')), 'members': []})
        if entry['po_id'] != po_id:
            raise GroupError(
                '{}: group {!r} carries two different po_id values ({!r} and {!r}). One '
                'group is one cell type -- split it, or make the ids agree.'.format(
                    path, gid, entry['po_id'], po_id))
        entry['members'].append(member)
    return GroupDocument(Path(path).name.split('_groups')[0], groups=groups,
                         dropped=dropped, ungrouped=ungrouped)


def check_parents(doc: GroupDocument, annotations: dict, allow_stale=False):
    """
    Refuse a grouping derived from annotation tables that have since changed.

    Same guard as `refilter`'s `parent_config_hash` and `decisions.check_parent`:
    re-running stage 1 can change which candidates exist, and a grouping naming
    a term that is no longer a candidate is not something to carry forward
    silently.
    """
    if not doc.parent_hashes:
        return
    stale = []
    for method, recorded in sorted(doc.parent_hashes.items()):
        path = annotations.get(method)
        if path is None or not Path(path).exists():
            continue
        current = file_hash(path)
        if current != recorded:
            stale.append('{} ({}... -> {}...)'.format(method, recorded[:12], current[:12]))
    if stale and not allow_stale:
        raise GroupError(
            '{} was written against different annotation tables: {}. Re-propose the '
            'grouping and redo the review, or pass allow_stale=True if the change is '
            'known to be irrelevant.'.format(doc.source, '; '.join(stale)))
