"""Opt-in typed generation and one local correction within the legacy call budget."""
from . import workflow, builder, typed_builder, typed_schema, local_repair, protocol


class Job(workflow.Job):
    def __init__(self, path, request):
        super().__init__(path, request, builder=typed_builder.Builder(request['contract'] == 'typed'))
        self.patching = False
        self.paths = []
        self.failed_object = None

    def task(self):
        task = super().task()
        if task is None:
            return None
        self.patching = False
        if self.error and self.request.get('local_repair'):
            try:
                self.failed_object = protocol.load_object(self.failed_text)
                self.paths = local_repair.allowed_paths(self.failed_object)
            except (ValueError, TypeError):
                self.paths = []
            if self.paths:
                self.patching = True
                task['_kind'] = 'local_repair'
                task['prompt'] = (self.request['prompt']+'\nLOCAL COMPILER ERROR (data): '+self.error
                    +'\nFAILED STEP (data): '+builder.dump(self.failed_object)
                    +'\nALLOWED REPLACEMENT PATHS: '+builder.dump(self.paths)
                    +'\nReturn {"edits":[{"path":"one allowed JSON pointer","value":NEW_VALUE}],"reason":"short reason"}. Replace only erroneous mathematical fields. No removals, new constraints, changed IDs or changed requirements. Original obligations remain mandatory. Use the typed catalog to select actual columns and explicit index mappings. The repaired complete step must compile; no silent correction is performed.')
        if self.request['contract'] == 'typed':
            task['_schema'] = typed_schema.schema(task['_kind'], self.request['question'],
                                                  len(self.request['variable_names']), self.paths)
        return task

    def accept(self, row):
        if self.patching and row.get('status') == 'generated':
            try:
                candidate = local_repair.apply(self.failed_object, row['response'], self.paths)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                self.records.append(row)
                self.fail('Local repair rejected: '+str(exc))
                return
            row = dict(row, response=builder.dump(candidate), repair_patch=row['response'])
        super().accept(row)
        if self.done and self.result:
            self.result['contract'] = self.request['contract']
            self.result['plan_audit'] = (self.state or {}).get('audit_notes', [])


def factory(path, request):
    return workflow.Job(path, request) if request.get('contract') == 'legacy' else Job(path, request)


def execute(generator, items, run, rank, batch_id):
    # The batch decoder uses the smallest output cap. Never let a nearly exhausted
    # request reduce another arm's available budget in this experiment.
    return workflow.execute(generator, items, run, rank, batch_id, job_factory=factory, isolate_budgets=True)
