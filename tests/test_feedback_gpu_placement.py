from types import SimpleNamespace as NS
import pytest
from experiments.feedback_study_v1.placement import verify_placement,DevicePlacementError


class Device:
    def __init__(self,kind,index=None):self.type,self.index=kind,index
    def __str__(self):return self.type+(f':{self.index}' if self.index is not None else '')


def tensor(kind='cuda',index=0,size=2):
    return NS(device=Device(kind,index),numel=lambda:size,element_size=lambda:2)


def model(params=None,buffers=None,mapping=None):
    return NS(named_parameters=lambda:params if params is not None else [('w',tensor())],
              named_buffers=lambda:buffers or [],hf_device_map=mapping)


@pytest.mark.parametrize('mapping',[None,{}, {'':'cuda:0'},{'':0}])
def test_gpu_residency_succeeds_without_optional_map(mapping):
    result=verify_placement(model(mapping=mapping),1)
    assert result['status']=='passed' and result['tensor_devices']=={'cuda:0':{'tensors':1,'bytes':4}}


def test_model_with_no_mapping_attribute_and_two_devices():
    m=model(params=[('a',tensor()),('b',tensor(index=1))]);del m.hf_device_map
    assert set(verify_placement(m,2)['tensor_devices'])=={'cuda:0','cuda:1'}


@pytest.mark.parametrize('device',['cpu','meta'])
def test_cuda_map_does_not_hide_actual_cpu_or_meta_weights(device):
    with pytest.raises(DevicePlacementError,match='parameters/w') as e:
        verify_placement(model(params=[('w',tensor(device,None))],mapping={'':'cuda:0'}),1)
    assert e.value.report['status']=='failed'


@pytest.mark.parametrize('mapping',[{'':'cpu'},{'':'disk'},{'':'meta'},{'':-1},{'':'cuda:2'},'invalid'])
def test_declared_offload_or_invalid_map_is_rejected(mapping):
    with pytest.raises(DevicePlacementError):verify_placement(model(mapping=mapping),1)


def test_cpu_buffer_rejected_but_empty_placeholder_is_not_offload():
    with pytest.raises(DevicePlacementError,match='buffers/cache'):
        verify_placement(model(buffers=[('cache',tensor('cpu',None))]),1)
    r=verify_placement(model(buffers=[('empty',tensor('cpu',None,0))]),1)
    assert r['empty_tensors_ignored']==1


def test_wrong_gpu_index_and_uninspectable_model_fail_closed():
    for m in (model(params=[]),model(params=[('w',tensor(index=1))]),NS(hf_device_map={'':'cuda:0'})):
        with pytest.raises(DevicePlacementError):verify_placement(m,1)


def test_resident_weights_do_not_hide_offload_hook():
    m=model();m.named_modules=lambda:[('layer',NS(_hf_hook=NS(hooks=[NS(offload=True)])))]
    with pytest.raises(DevicePlacementError,match='offload hook'):verify_placement(m,1)
