"""Bounded atomic replacements of existing mathematical fields; no silent fixes."""
import copy
from .protocol import load_object


def allowed_paths(obj):
    paths = []
    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                pointer = path+'/'+key.replace('~', '~0').replace('/', '~1')
                # Routing/evidence/IDs and list membership are immutable in local repair.
                if key in ('linear', 'quadratic', 'sense', 'a', 'rhs', 'op', 'expr', 'lower', 'upper', 'terms'):
                    paths.append(pointer)
                elif key not in ('requirements', 'grounding', 'revision_notes', 'audit_notes'):
                    visit(child, pointer)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, path+'/'+str(i))
    visit(obj, '')
    return paths


def apply(original, text, paths):
    patch = load_object(text)
    if set(patch) != {'edits', 'reason'} or not isinstance(patch['reason'], str):
        raise ValueError('Local repair requires edits and reason')
    edits = patch['edits']
    if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
        raise ValueError('Local repair requires 1..8 edits')
    seen = []
    result = copy.deepcopy(original)
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'path', 'value'} or edit['path'] not in paths:
            raise ValueError('Local repair targeted an immutable or absent path')
        path = edit['path']
        if any(path == p or path.startswith(p+'/') or p.startswith(path+'/') for p in seen):
            raise ValueError('Overlapping or duplicate local repairs')
        seen.append(path)
        parts = [v.replace('~1', '/').replace('~0', '~') for v in path.split('/')[1:]]
        parent = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
        parent[key] = copy.deepcopy(edit['value'])
    return result
