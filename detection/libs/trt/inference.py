try:
    import tensorrt
    import pycuda.autoinit
    import pycuda.driver as cuda
except ImportError:
    raise ImportError("Failed to load tensorrt")

import os
import numpy
from ...models.enum import DataAttribute
from typing import Tuple, List, Dict, NewType
from ...models.error import AlreadyInitializedError, UninitializedError

__interpreter_dict = {}
INPUT = 'input'
OUTPUT= 'output'
INTERPRETER = 'interpreter'

input_attribute = NewType('input_attribute', Dict[str, DataAttribute])
output_attribute = NewType('output_attribute', Dict[str, DataAttribute])

def model_initialize(cls, **kwargs)->None:
    """Todo

    Raises
    ------
    AlreadyInitializedError
        Todo
    """
    if is_init(cls) is True:
        raise AlreadyInitializedError(cls.__name__)
    
    inputs = {}
    outputs = {}
    bindings = []
    
    class HostDeviceMem(object):
        def __init__(self, cpu_mem, gpu_mem):
            self.cpu = cpu_mem
            self.gpu = gpu_mem
    
    TRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    runtime = tensorrt.Runtime(TRT_LOGGER)
    
    
    with open(os.path.join(os.path.dirname(__file__), "model.engine"), "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    input_attribute, output_attribute = model_input_output_attributes(engine)
    
    
    for binding in engine:
        size = tensorrt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = tensorrt.nptype(engine.get_binding_dtype(binding))
        cpu_mem = cuda.pagelocked_empty(size, dtype)
        gpu_mem = cuda.mem_alloc(cpu_mem.nbytes)
        bindings.append(int(gpu_mem))
        if engine.binding_is_input(binding):
            inputs[binding] = HostDeviceMem(cpu_mem, gpu_mem)
        else:
            outputs[binding] = HostDeviceMem(cpu_mem, gpu_mem)

    __interpreter_dict[cls] = {
        INTERPRETER: context,
        INPUT: input_attribute,
        OUTPUT: output_attribute,
        "inputs":inputs,
        "outputs":outputs,
        "bindings":bindings,
        "stream":cuda.Stream()
    }

def model_input_output_attributes(engine:tensorrt.ICudaEngine)->Tuple[input_attribute, output_attribute]:
    """Todo

    Parameters
    ----------
    engine : tensorrt.ICudaEngine
        Todo

    Returns
    -------
    Tuple[input_attribute, output_attribute]
        Todo
    """
    inputs = {}
    outputs = {}
    
    for binding in engine:
        binding_index = engine.get_binding_index(binding)
        data_attribute = DataAttribute()
        data_attribute.name = engine.get_binding_name(binding_index)
        data_attribute.location = None
        data_attribute.shape = tuple(engine.get_binding_shape(binding))
        data_attribute.dtype = engine.get_binding_dtype(binding).name
        if engine.binding_is_input(binding):
            data_attribute.format = engine.get_binding_format(binding_index).name
            inputs[data_attribute.key] = data_attribute
        else:
            outputs[data_attribute.key] = data_attribute
        
    return inputs, outputs

def model_finalize(cls:str)->None:
    """Todo

    Raises
    --------
    UninitializedError
        초기화 되지 않은 class에 model_inference를 요청하면 에러를 출력합니다.

    """
    if is_init(cls) is False:
        raise UninitializedError
    __interpreter_dict.pop(cls)

def model_inference(cls:str, preprocess_result: Dict[int, numpy.ndarray], **kwargs)\
    ->Dict[int, numpy.ndarray]:
    """Todo

    Returns
    -------
    Dict[int, numpy.ndarray]
        Todo

    Raises
    ------
    UninitializedError
        Todo
    """
    interpreter_dict = __interpreter_dict.get(cls, None)
    if interpreter_dict is None:
        raise UninitializedError

    inputs = interpreter_dict.get('inputs')
    outputs = interpreter_dict.get('outputs')
    bindings = interpreter_dict.get('bindings')
    stream = interpreter_dict.get('stream')
    context = interpreter_dict.get(INTERPRETER)
    output = interpreter_dict.get(OUTPUT)
    
    for k,v in preprocess_result.items():
        inputs[k].cpu = v.ravel()
    
    [cuda.memcpy_htod_async(inp.gpu, inp.cpu, stream) for inp in inputs.values()]
    context.execute_async_v2(bindings, stream.handle, None)
    [cuda.memcpy_dtoh_async(out.cpu, out.gpu, stream) for out in outputs.values()]
    stream.synchronize()
    
    output_dict = {}
    for output_name in iter(output):
        output_dict[output_name] = outputs[output_name].cpu
    return output_dict

def is_init(cls)->bool:
    """입력 받은 클래스가 초기화 되었는지 확인합니다.
    
    Arguments
    --------
    cls: Class
        초기화 상태를 확인하기 위한 클래스

    Returns
    -------
    bool:
        전역변수 __interpreter_dict에 cls이름과 같은 key가 있다면 True를 출력합니다.

    """
    return True if __interpreter_dict.get(cls, None) is not None else False
    
def get_input_output_attributes(cls):
    """Todo"""
    dictionary = __interpreter_dict.get(cls, None)
    if dictionary is None:
        return None, None
    return dictionary.get(INPUT, None), dictionary.get(OUTPUT, None)
