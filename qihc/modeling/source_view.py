"""Compact, lossless numeric reference view; full provenance stays in the resolver."""
import copy
import json

from .sources import catalog


COLUMNS = ('id', 'kind', 'start', 'end', 'path', 'value')


def prompt_catalog(question):
    # Keep every ID, exact value, location and JSON pointer, including duplicate
    # values at distinct locations. Do not mutate the cached audit catalog.
    return {'columns': list(COLUMNS),
            'rows': [[copy.deepcopy(row[key]) for key in COLUMNS] for row in catalog(question)]}


def render_catalog(question):
    return ('Rows use the listed columns. start/end are zero-based character offsets '
            '[start,end) in the original question; path is a JSON pointer within that span. '
            'Use the id in source references. All values are exact. Hashes and repeated '
            'context are retained by the verifier, not needed in your output.\n'
            + json.dumps(prompt_catalog(question), ensure_ascii=False, separators=(',', ':')))
