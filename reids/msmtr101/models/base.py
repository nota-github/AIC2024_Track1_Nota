import abc
import cv2
import logging
import numpy
from typing import Union, Any, Dict
from ..libs.trt.inference import model_inference, model_initialize, \
    model_finalize, is_init, get_input_output_attributes

from .enum import DataAttribute
from .error import unsupported_initialize, unsupported_finalize, UninitializedError
from . import log
logger = logging.getLogger(log.__name__)

class Basemodel(metaclass=abc.ABCMeta):
    """Todo
    """
    def __new__(cls): 
        if is_init(cls) is False:
            raise UninitializedError
        obj = super().__new__(cls)
        setattr(obj, "initialize", unsupported_initialize)
        setattr(obj, "finalize", unsupported_finalize)
        return obj

    @property
    def inputs(cls)->Union[None, Dict[int, DataAttribute]]:
        input, _ = get_input_output_attributes(cls.__class__)
        return input
    
    @property
    def outputs(cls)->Union[None, Dict[int, DataAttribute]]:
        _, output = get_input_output_attributes(cls.__class__)
        return output
    
    @classmethod 
    def initialize(cls, **kwargs)->None:
        """Todo"""
        model_initialize(cls=cls, **kwargs)

    @classmethod 
    def finalize(cls, **kwargs)->None:
        """Todo"""
        model_finalize(cls=cls, **kwargs)
    
    def preprocess(self, input_data: Union[str, Any])->Dict[Union[str, int], numpy.ndarray]:
        """A function that preprocesss input data as defined in the input layer of the model.
        Override this function in the child class. It into input data that fits the model.
        
        Parameters
        ----------
        input_data : Union[str, Any]
            The value of input_data recevies the model path or model raw data.

        Returns
        -------
        Dict[Union[str, int], numpy.ndarray]
            Returns the image raw data dictionary.
            The key of the Dictionary is the name or location of the input layer 
            and value is raw data of image
        """
        if isinstance(input_data,str) is True:
            return {next(iter(self.inputs)):cv2.imread(input_data)}
        else:
            return {next(iter(self.inputs)):input_data}
    
    def postprocess(self, inference_result: numpy.ndarray)->Any:
        """Todo"""
        return inference_result

    def run(self, input_data:Any, **kwargs):
        """Todo"""
        pre_result = self.preprocess(input_data)
        inference_result = model_inference(cls=self.__class__, preprocess_result=pre_result)
        post_result = self.postprocess(inference_result)
        return post_result