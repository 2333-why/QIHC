"""Process-local version adapter; no E004-1.1 source files are changed."""
import copy
import hashlib
import os
from pathlib import Path
from experiments.feedback_study_v1 import common

EXPERIMENT = 'E004-1.2'
HERE = Path(__file__).resolve().parent
_fingerprint = common.fingerprint
_installed = False


def default_root():
    return Path(os.environ.get('QIHC_PILOT_RUN', str(common.BASE/'results/E004-1.2_feedback_pilot')))


def fingerprint():
    h = hashlib.sha256(_fingerprint().encode())
    paths = list(HERE.glob('*')) + [common.ROOT/'experiments/feedback_comparison_v1/report.py']
    for p in sorted(paths):
        if p.suffix in ('.py', '.json', '.sh'):
            h.update(p.relative_to(common.ROOT).as_posix().encode())
            h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def bounded_schema(schema):
    """Replace recursive expression references with a finite, arity-correct DAG."""
    out = copy.deepcopy(schema)
    index = copy.deepcopy(out['$defs']['expr']['anyOf'][0]['properties']['var'])
    leaf = {'type': 'object', 'properties': {'var': index}, 'required': ['var'], 'additionalProperties': False}
    defs = {'expr_d0': leaf}
    for depth in range(1, 5):
        variants = [copy.deepcopy(leaf)]
        for op, lo, hi in [('not', 1, 1), ('and', 2, 8), ('or', 2, 8), ('implies', 2, 2), ('iff', 2, 2)]:
            variants.append({'type': 'object', 'properties': {
                'op': {'const': op}, 'args': {'type': 'array', 'items': {'$ref': f'#/$defs/expr_d{depth-1}'},
                                             'minItems': lo, 'maxItems': hi}},
                'required': ['op', 'args'], 'additionalProperties': False})
        defs[f'expr_d{depth}'] = {'anyOf': variants}
    defs['expr'] = {'$ref': '#/$defs/expr_d4'}
    out['$defs'].update(defs)

    def visit(node):
        if isinstance(node, dict):
            for name, prop in node.get('properties', {}).items():
                if name == 'hypotheses': prop['maxItems'] = 1
                if name == 'edits': prop['maxItems'] = 4
                if name in ('quote', 'reason', 'notes') and prop.get('type') == 'string':
                    prop['maxLength'] = 320
            for value in node.values(): visit(value)
        elif isinstance(node, list):
            for value in node: visit(value)
    visit(out)
    return out


def install():
    global _installed
    if _installed:
        return
    from experiments.feedback_study_v1 import protocol, runtime, gpu, prepare, report as legacy_report
    from .report import report
    original_validate = common.validate
    original_schemas = protocol.schemas
    original_request = protocol.build_request

    def validate(root):
        m = common.s.load(root/'manifest.json')
        if m.get('experiment_id') != EXPERIMENT:
            raise ValueError('Pilot entry refuses non-E004-1.2 run directories')
        for name, key in [('evaluation/cases.CPU_ONLY.json','evaluation_digest'),
                          ('evaluation/e003_baselines.json','baselines_digest')]:
            if common.s.digest(root/name) != m[key]:
                raise ValueError('Pilot frozen evidence identity changed')
        return original_validate(root)

    def schemas(question, names):
        return tuple(bounded_schema(syntax) for syntax in original_schemas(question, names))

    def build_request(case, state, stage, evidence=None):
        request = original_request(case, state, stage, evidence)
        request['prompt'] += (
            '\nPILOT OUTPUT CONTRACT: Be concise. At most one alternative hypothesis and four edits. '
            'Use empty hypotheses/edits when no source-supported correction is needed. '
            'Avoid redundant nested Boolean expressions; maximum operator depth is four. '
            'All indices are zero-based variable indices. A data table row is not necessarily a variable: '
            'check the number and meaning of rows before using row_index. Distinguish one summed constraint '
            'from separate constraints for each row. Never fix an out-of-range index by dropping a requirement. '
            'When rebuilding a failed model, address the reported compiler error instead of repeating its binding. '
            'Quote/reason/notes must each fit 320 characters.')
        return request

    common.fingerprint = fingerprint
    common.default_root = default_root
    common.validate = validate
    runtime.validate = gpu.validate = validate
    runtime.default_root = gpu.default_root = default_root
    protocol.schemas = prepare.schemas = schemas
    protocol.build_request = runtime.build_request = prepare.build_request = build_request
    runtime.ENTRY = 'experiments.feedback_pilot_v1'
    legacy_report.report = report
    _installed = True
