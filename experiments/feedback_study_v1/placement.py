"""Verify resident model tensors, without relying on optional loader metadata."""
from collections import defaultdict


class DevicePlacementError(RuntimeError):
    def __init__(self, report):
        self.report=report
        super().__init__('GPU placement verification failed: '+str(report['issues'])+
                         '; tensor devices='+str(report['tensor_devices']))


def verify_placement(model, visible_count):
    mapping=getattr(model,'hf_device_map',None)
    report={'status':'checking','hf_device_map':{str(k):str(v) for k,v in mapping.items()} if isinstance(mapping,dict) else None,
            'visible_cuda_devices':visible_count,'tensor_devices':{},'parameters':0,'buffers':0,
            'empty_tensors_ignored':0,'issues':[]}
    issues=report['issues']; devices=defaultdict(lambda:{'tensors':0,'bytes':0})
    def fail(message):
        if len(issues)<20:issues.append(message)
    if visible_count<=0:fail('No visible CUDA device')
    # A declared offload map is not silently ignored even if some tensors are
    # temporarily on CUDA. Missing metadata, however, says nothing about fit.
    if mapping is not None and not isinstance(mapping,dict):fail('Malformed hf_device_map')
    for name,device in (mapping or {}).items() if isinstance(mapping,dict) else ():
        value=str(device)
        if value.isdigit():index=int(value)
        elif value=='cuda':index=0
        elif value.startswith('cuda:') and value[5:].isdigit():index=int(value[5:])
        else:
            fail(f'Declared offload/non-CUDA placement {name}: {value}');continue
        if not 0<=index<visible_count:fail(f'Declared CUDA index outside assignment {name}: {value}')
    for kind,method in (('parameters','named_parameters'),('buffers','named_buffers')):
        fn=getattr(model,method,None)
        if not callable(fn):
            fail('Cannot inspect model '+method);continue
        for name,tensor in fn():
            if tensor.numel()==0:
                report['empty_tensors_ignored']+=1;continue
            report[kind]+=1
            device=tensor.device;label=str(device)
            devices[label]['tensors']+=1;devices[label]['bytes']+=int(tensor.numel()*tensor.element_size())
            if device.type!='cuda':fail(f'{kind}/{name} is on {label}')
            elif device.index is None or not 0<=device.index<visible_count:
                fail(f'{kind}/{name} has unverified CUDA index: {label}')
    if not report['parameters']:fail('No nonempty parameters inspected')
    modules=getattr(model,'named_modules',lambda:[])()
    for name,module in modules:
        hooks=[getattr(module,'_hf_hook',None)];seen=set()
        while hooks:
            hook=hooks.pop()
            if hook is None or id(hook) in seen:continue
            seen.add(id(hook))
            if getattr(hook,'offload',False):fail(f'Active offload hook at {name}')
            hooks.extend(getattr(hook,'hooks',()) or ())
    report['tensor_devices']=dict(devices)
    report['status']='failed' if issues else 'passed'
    if issues:raise DevicePlacementError(report)
    return report
