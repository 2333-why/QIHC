"""Opt-in canonical assembly and one checkpointed semantic review, same total cap."""
import copy
from . import workflow, typed_workflow, semantic_review, protocol
from .canonical_builder import Builder


class Job(typed_workflow.Job):
    def __init__(self, path, request):
        super().__init__(path, request)
        if request.get('canonical_ids'):
            self.builder = Builder()
        self.reviewing = False
        self.initial = None

    def task(self):
        if self.reviewing:
            remaining = self.builder.TOTAL_OUTPUT_TOKENS-self.tokens
            if remaining < 128 or len(self.records) >= 8:
                self.rollback('skipped_budget', 'Shared 16384-token/8-call cap exhausted')
                return None
            return {'prompt': semantic_review.prompt(self.request, self.initial['response']),
                    'question': self.request['question'], '_kind': 'semantic_review',
                    '_max_new_tokens': min(8192, remaining),
                    '_schema': semantic_review.schema(self.request['question'], len(self.request['variable_names']))}
        task = super().task()
        # Same initial prompt/cap for exact paired comparisons. Review uses only
        # unused budget, never extends the 8-call / 16384-output-token ceiling.
        return task

    def rollback(self, status, error):
        self.result = copy.deepcopy(self.initial)
        self.result['semantic_review'] = {'status': status, 'error': str(error)}
        self.result['initial_response'] = self.initial['response']
        self.done = True
        self.reviewing = False

    def accept(self, row):
        if self.reviewing:
            self.records.append(row)
            try:
                if row.get('status') != 'generated':
                    raise ValueError('Review unfinished: '+str(row.get('finish_reason', row.get('error'))))
                response, audit = semantic_review.validate_review(protocol.load_object(row['response']),
                    self.initial['response'], self.request['question'], self.request['variable_names'])
            except (ValueError, TypeError, KeyError, OverflowError, IndexError) as exc:
                self.rollback('rejected_preserved_initial', exc)
            else:
                self.result = dict(self.initial, response=response, initial_response=self.initial['response'], semantic_review=audit)
                # The staged state describes the initial model, never the reviewed replacement.
                if 'builder_state' in self.result:
                    self.result['initial_builder_state'] = self.result.pop('builder_state')
                self.done = True
                self.reviewing = False
            return
        super().accept(row)
        if self.done and self.result.get('mathematical_validity') == 'valid':
            if self.request.get('canonical_ids') and self.state is not None:
                self.result['constraint_id_map'] = self.builder.canonical_state(self.state)[1]
            if self.request.get('semantic_review'):
                self.initial = copy.deepcopy(self.result)
                self.done = False
                self.reviewing = True


def factory(path, request):
    if request.get('canonical_ids') or request.get('semantic_review'):
        return Job(path, request)
    return typed_workflow.factory(path, request)


def execute(generator, items, run, rank, batch_id):
    return workflow.execute(generator, items, run, rank, batch_id, job_factory=factory, isolate_budgets=True)
