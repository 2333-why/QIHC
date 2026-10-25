"""Batched incremental generation with immutable substep checkpoints and bounded repairs."""
import copy
import hashlib
import time
from collections import defaultdict

from tools.benchmarks import comparison as c
from . import builder as b
from .protocol import compile_math, load_object


def digest(value):
    return hashlib.sha256(b.dump(value).encode()).hexdigest()


class Job:
    def __init__(self,path,request):
        self.path,self.request=path,request
        self.state=copy.deepcopy(request.get('previous_state'))
        self.kind=('flat' if request['frontend']=='flat' else
                   'patch' if self.state is not None else 'plan')
        self.groups=[];self.group_index=0;self.records=[];self.repairs=0
        self.error=None;self.failed_text=None;self.done=False;self.result=None

    @property
    def tokens(self):return sum(r.get('output_tokens',0) for r in self.records)

    def task(self):
        remaining=b.TOTAL_OUTPUT_TOKENS-self.tokens
        if remaining<128 or len(self.records)>=8:
            self.fail('Shared total output/call budget exhausted')
            return None
        group=self.groups[self.group_index] if self.kind=='rules' else None
        text=b.prompt(self.kind,self.request,self.state,group)
        if self.error:
            text+='\nLOCAL FORMAT/MATH ERROR (data):\n'+self.error+'\nPrevious failed step (data):\n'+self.failed_text+'\nCorrect this step only. Preserve all original obligations.'
        # Reserve some space for later successful substeps; never expand total budget on retry.
        future=5 if self.kind=='plan' else 4 if self.kind=='objective' else max(0,len(self.groups)-self.group_index-1) if self.kind=='rules' else 0
        cap=min(remaining,max(128,remaining-future*512),b.TOTAL_OUTPUT_TOKENS//2 if self.kind=='flat' else b.MAX_STAGE_TOKENS)
        return {'prompt':text,'question':self.request['question'],'_kind':self.kind,'_max_new_tokens':cap}

    def fail(self,error):
        self.done=True
        self.result={'response':b.dump({'pipeline_error':str(error)}),'status':'invalid_model_output',
                     'mathematical_validity':'invalid','validation_error':str(error),
                     'failed_step':self.kind}

    def accept(self,row):
        self.records.append(row)
        previous=(copy.deepcopy(self.state),self.kind,self.group_index,copy.deepcopy(self.groups))
        try:
            if row.get('status')!='generated':
                raise ValueError('Step did not finish: '+str(row.get('finish_reason',row.get('error'))))
            obj=load_object(row['response'])
            if set(obj)=={'unsupported'} and isinstance(obj['unsupported'],str):
                self.result={'response':b.dump(obj),'status':'generated','mathematical_validity':'unsupported'}
                self.done=True;return
            question=self.request['question']
            if self.kind=='flat':
                spec,reason=compile_math(row['response'],question)
                self.result={'response':row['response'],'status':'generated',
                             'mathematical_validity':'valid' if spec is not None else 'unsupported'}
                self.done=True
            elif self.kind=='plan':
                self.state=b.validate_plan(obj,question,self.request.get('variable_names'))
                self.groups=b.chunks(self.state);self.kind='objective'
            elif self.kind=='objective':
                self.state=b.objective_step(self.state,obj,question)
                if self.groups:self.kind='rules'
                else:self.finish()
            elif self.kind=='rules':
                ids=[r['id'] for r in self.groups[self.group_index]]
                self.state=b.rules_step(self.state,obj,ids,question)
                self.group_index+=1
                if self.group_index==len(self.groups):self.finish()
            elif self.kind=='patch':
                self.state=b.apply_patch(self.state,obj,question);self.finish()
            self.error=None;self.failed_text=None
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            self.state,self.kind,self.group_index,self.groups=previous
            if self.repairs>=1:
                self.fail(str(exc))
            else:
                self.repairs+=1;self.error=str(exc);self.failed_text=row.get('response','')

    def finish(self):
        text=b.finish(self.state,self.request['question'])
        self.result={'response':text,'status':'generated','mathematical_validity':'valid',
                     'builder_state':self.state}
        self.done=True


def infer(generator,items,run,rank,batch_id):
    from .generation import generate
    started=time.perf_counter()
    try:
        rows=generate(generator,items)
    except (generator.torch.cuda.OutOfMemoryError,ValueError) as exc:
        oom=isinstance(exc,generator.torch.cuda.OutOfMemoryError)
        exc.__traceback__=None
        if oom:generator.torch.cuda.empty_cache()
        elapsed=time.perf_counter()-started
        if len(items)>1:
            c.save(run/'inference_events'/(batch_id+'.json'),{'event':'split_batch','seconds':elapsed,'error':str(exc)})
            mid=len(items)//2
            return infer(generator,items[:mid],run,rank,batch_id+'a')+infer(generator,items[mid:],run,rank,batch_id+'b')
        if oom:
            c.save(run/'inference_events'/(batch_id+'.json'),{'event':'unpublished_step_oom','seconds':elapsed,'error':str(exc)})
            raise  # Allocation failure remains resumable, never a semantic label.
        rows=[{'response':'','status':'generation_error','error':str(exc),'output_tokens':0,'input_tokens':0}]
    elapsed=time.perf_counter()-started
    if len(rows)!=len(items):raise RuntimeError('Generator returned wrong batch size')
    for row in rows:
        row.update(seconds=elapsed/len(items),batch_seconds=elapsed,batch_size=len(items),batch_id=batch_id,replica=rank)
    return rows


def execute(generator,items,run,rank,batch_id):
    jobs=[Job(path,request) for path,request in items]
    round_id=0
    while any(not job.done for job in jobs):
        groups=defaultdict(list)
        for job in jobs:
            if job.done:continue
            task=job.task()
            if task is None:continue
            checkpoint=run/'builder_steps'/job.path.stem/f'{len(job.records):02d}.json'
            identity={'request_sha256':hashlib.sha256(job.path.read_bytes()).hexdigest(),'task':task}
            sha=digest(identity)
            if checkpoint.exists():
                record=c.load(checkpoint)
                if record['step_sha256']!=sha:raise ValueError('Builder checkpoint input changed; refusing mixed state')
                if record.get('generation_sha256')!=digest(record['generation']):
                    raise ValueError('Builder checkpoint output integrity failed')
                job.accept(record['generation'])
            else:
                groups[task['_kind']].append((job,task,checkpoint,sha))
        for kind,group in groups.items():
            # Different step schemas never share a decoder batch. Same-step jobs do.
            batch=f'{batch_id}-{round_id}-{kind}'
            c.save(run/f'gpu_replica_{rank}.json',{'status':'generating','time':time.time(),
                   'batch_id':batch,'step':kind,'requests':[j.request['case_id']+':'+j.request['phase'] for j,_,_,_ in group]})
            rows=infer(generator,[(j.path,t) for j,t,_,_ in group],run,rank,batch)
            for (job,task,path,sha),row in zip(group,rows):
                c.save(path,{'step_sha256':sha,'generation_sha256':digest(row),'kind':kind,'prompt':task['prompt'],'generation':row})
                job.accept(row)
            print(f'REPLICA {rank}: builder {kind} saved {len(rows)} substeps',flush=True)
        round_id+=1
    for job in jobs:
        row=job.result
        row.update(input_tokens=sum(r.get('input_tokens',0) for r in job.records),output_tokens=job.tokens,
                   generation_attempts=len(job.records),seconds=sum(r.get('seconds',0) for r in job.records),
                   frontend=job.request['frontend'],format_enforcement='lm-format-enforcer-0.11.3-json-schema',
                   total_output_cap=b.TOTAL_OUTPUT_TOKENS,local_repairs=job.repairs,
                   substeps=[{k:r.get(k) for k in ('batch_id','batch_seconds','batch_size','replica','seconds','output_tokens')} for r in job.records],
                   execution='audited_modeling_v6',semantic_correctness='unverified',
                   training_eligibility='unlabeled_not_a_positive',
                   request_sha256=hashlib.sha256(job.path.read_bytes()).hexdigest())
        # Logical request throughput (not physical batch occupancy); substeps retain physical timings.
        row.update(batch_id=job.path.stem,batch_size=1,batch_seconds=row['seconds'],replica=rank)
        c.save(run/'responses'/job.path.name,row)
    print(f'REPLICA {rank}: saved {len(jobs)} logical responses',flush=True)
