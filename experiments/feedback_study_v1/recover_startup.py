"""Audited one-time upgrade ONLY for the known pre-inference startup failure.

No finished experiment is relabeled. Existing preparation, queue, and logs stay
in place. A durable before/after receipt makes interrupted migration replayable.
"""
import argparse
import copy
import hashlib
from contextlib import ExitStack
from pathlib import Path
import time
from .common import s, ROOT, default_root, fingerprint, commit, versions, validate
from qihc.experiment_runtime.identity import model_identity

OLD_SOURCE_HASH='62fd9a27f432375104ae6622e51d4ddcf66b454f6056d28f7bc89de34c4bb53b'
# Excludes this receipt writer to avoid a self-referential hash; includes every
# model, algorithm, runtime and launch file. A future code change cannot silently
# reuse today's compatibility assertion.
COMPATIBLE_PATCH_HASH='aa4d26c6ddb4d09c92ea6ffd9efa3b713a56ab755a00055251a69776775e1a9d'


def patch_fingerprint():
    h=hashlib.sha256()
    for folder in ('qihc/modeling','qihc/generic','qihc/experiment_runtime','experiments/feedback_study_v1'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.suffix in ('.py','.json','.sh') and path!=Path(__file__).resolve():
                h.update(path.relative_to(ROOT).as_posix().encode())
                h.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return h.hexdigest()


def check_preparation(root, manifest):
    if s.value_digest(manifest['config'])!=manifest['config_digest']:
        raise ValueError('Prepared configuration changed')
    for path,field in (('cases.public.json','public_digest'),('evaluation/cases.CPU_ONLY.json','evaluation_digest')):
        if s.digest(root/path)!=manifest[field]:
            raise ValueError('Prepared data changed: '+path)
    if not manifest.get('smoke'):
        if versions()!=manifest['dependencies']:
            raise ValueError('Prepared environment changed; restore it before recovery')
        if model_identity(Path(manifest['model']))!=manifest['model_files']:
            raise ValueError('Prepared weight metadata changed')


def recover(root):
    if patch_fingerprint()!=COMPATIBLE_PATCH_HASH:
        raise ValueError('Code differs from the audited device-check patch; recovery compatibility not established')
    receipt_path=root/'recovery/startup-devicecheck-v1.json'
    with ExitStack() as stack:
        stack.enter_context(s.worker_lock(root/'launch.lock'))
        for rank in range(4):stack.enter_context(s.worker_lock(root/f'gpu-{rank}.lock'))
        m=s.load(root/'manifest.json');pf=s.load(root/'preflight.json');current=fingerprint()
        check_preparation(root,m)
        if receipt_path.exists():
            receipt=s.load(receipt_path)
            if receipt['after_manifest']['source_hash']!=current:
                raise ValueError('Recovery belongs to another code revision')
            if m not in (receipt['before_manifest'],receipt['after_manifest']) or pf not in (receipt['before_preflight'],receipt['after_preflight']):
                raise ValueError('Recovery receipt does not match prepared files')
            if m==receipt['after_manifest'] and pf==receipt['after_preflight']:
                validate(root)
                return {'status':'already_recovered','receipt':str(receipt_path)}
        else:
            # Fresh preparation made with this code needs no compatibility upgrade.
            if m['source_hash']==current:
                if pf.get('source_hash')!=current or pf.get('status')!='passed':
                    raise ValueError('Current source lacks a passing CPU preflight')
                validate(root)
                return {'status':'current_preparation','receipt':None}
            if m['source_hash']!=OLD_SOURCE_HASH:
                raise ValueError('Only the audited d7411dd/521daee startup revision can be migrated')
            if pf.get('source_hash')!=m['source_hash'] or pf.get('status')!='passed':
                raise ValueError('Original CPU preflight did not pass')
            receipt=None
        # This narrow migration is forbidden once any inference attempt or
        # search trajectory exists, including failures with unknown token cost.
        for folder in ('responses','inference_events','states','rounds','cpu_execution_events',
                       'evaluation/scores','evaluation/jobs','cpu_jobs','cpu_results'):
            if any((root/folder).rglob('*.json')):
                raise ValueError('Not a pre-inference startup failure; preserve run and use a separate recovery plan: '+folder)
        if (root/'acceptance.json').exists():
            raise ValueError('An acceptance artifact exists; cannot relabel this run')
        if receipt is None:
            after=copy.deepcopy(m);after.update(source_hash=current,code_commit=commit(),
                startup_recovery={'from_source_hash':m['source_hash'],'receipt':'recovery/startup-devicecheck-v1.json'})
            new_pf=copy.deepcopy(pf);new_pf.update(source_hash=current,
                reused_cpu_preflight_from_source_hash=pf['source_hash'],
                recovery_scope='Device-residency check/diagnostics only; tokenizer, schema, data and algorithm unchanged. Not a new GPU acceptance.')
            receipt={'time':time.time(),'before_manifest':m,'after_manifest':after,
                     'before_preflight':pf,'after_preflight':new_pf,
                     'reason':'Known startup guard conflated absent hf_device_map with offload; no inference or trajectories exist'}
            s.save(receipt_path,receipt)
        # The receipt is committed first; interruption between these two atomic
        # writes is repaired on the next invocation without losing old metadata.
        s.save(root/'manifest.json',receipt['after_manifest'])
        s.save(root/'preflight.json',receipt['after_preflight'])
        validate(root)
        return {'status':'recovered_pre_inference','receipt':str(receipt_path)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=default_root())
    print(recover(p.parse_args().run),flush=True)
