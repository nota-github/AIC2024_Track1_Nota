from enum import Enum
from typing import List, Tuple, Union

class EnumInputNodeShapeFormat(Enum):
    '''input node shape format의 열거형 집합입니다.
    포맷은 [n, c, h, w]의 조합으로 구성되어 있습니다.
    
    각 알파벳이 의미하는 값은 다음과 같습니다.
    N: number of images in the batch. 만약 'N'이 없다면 싱글 이미지를 뜻합니다.
    C: number of channels of the image
    H: height of the image
    W: width of the image 

    Raises
    ----------
    KeyError
        멤버에 없는 값을 받을 경우 에러를 출력합니다.

    See Also
    ----------
    openvino foramt: 
    https://docs.openvino.ai/latest/enumInferenceEngine_1_1Layout.html

    tensorrt format: 
    https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html
    '''
    """_summary_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    KeyError
        _description_
    """
    NCHW = 'nchw'
    NCWH = 'ncwh'
    NHWC = 'nhwc'
    NWHC = 'nwhc'
    
    CHW = NCHW
    CWH = NCWH
    HWC = NHWC
    WHC = NWHC
    
    # tensorrt format
    LINEAR = NCHW
    CHW2 = NCHW
    HWC8 = NCHW
    CHW4 = NCHW
    CHW16 = NCHW
    CHW32 = NCHW

    UNKNOWN = 'unknown'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value):
        value = str(value).upper()
        try:
            return cls[value]
        except KeyError:
            msg = f"{cls.__name__} expected {', '.join(list(cls.__members__.keys()))} but got `{value}`"
            raise KeyError(msg)
            


class EnumNodeRawDataType(Enum):
    '''node raw data type의 열거형 집합입니다.

    Raises
    ----------
    KeyError
        멤버에 없는 값을 받을 경우 에러를 출력합니다.

    See Also
    ----------
    openvino data type: 
    https://docs.openvino.ai/2021.3/classInferenceEngine_1_1Precision.html

    tensorrt data type: 
    https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/DataType.html
    '''
    # numpy data type
    FLOAT16='float16'
    FLOAT32='float32'
    FLOAT64='float64'
    INT8='int8'
    INT16='int16'
    INT32='int32'
    INT64='int64'
    UINT8='uint8'
    UINT16='uint16'
    UINT32='utin32'
    UINT64='uint64'
    #BOOL='bool_'

    # openvino data type
    FP16=FLOAT16
    FP32=FLOAT32
    FP64=FLOAT64
    I8=INT8
    I16=INT16
    I32=INT32
    I64=INT64
    U8=UINT8
    U16=UINT16
    U32=UINT32
    U64=UINT64

    # tensorrt data type
    FLOAT=FLOAT32
    HALF=FLOAT16

    UNKNOWN = 'unknown'

    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def _missing_(cls, value):
        value = str(value).upper()
        try:
            return cls[value]
        except KeyError:
            msg = f"{cls.__name__} expected {', '.join(list(cls.__members__.keys()))} but got `{value}`"
            raise KeyError(msg)

class DataAttribute():
    def __init__(self):
        """Node의 속성 정보를 저장하는 class입니다. 각 framework마다 속성 표현 방식이 다르기 때문에
        각 속성의 데이터를 변경하는 과정이 포함돼 있습니다.
        
        Attributes
        ----------
        attributes: lists
            DataAttribute에서 저장하는 속성들의 리스트 입니다.

        _shape: Union[None, Tuple]
            Node의 shape 정보

        _dtype: Union[None, str]
            Node에 저장된 raw data의 type 정보

        _format: Union[None, str]
            Node의 shape의 format 정보

        _location: Union[None, int]
            Node의 위치 정보
        
        _name: Union[None, str]
            Node의 이름 정보

        _height: Union[None, int]
            Node shape에서 height의 정보

        _width: Union[None, int]
            Node shape에서 width의 정보

        
        Methods
        -------
        get_props(): lists
            이 클래스의 property 객체들을 list로 묶어서 출력해줍니다.

        Examples
        --------
        >>> data = DataAttribute()
        >>> data.shape = (1,3,640,640)
        >>> data.dtype = 'float32'
        >>> data.format = 'nchw'
        >>> data.name = 'input'
        >>> data.location = 1
        >>> print(data.width, data.height)

        """
        self.attributes = self.get_props()
        for i in self.attributes:
            setattr(self, f'_{i}', None)
    def __dir__(self):
        return self.attributes
    def __iter__(self):
        for i in dir(self):
            yield i, getattr(self, i)
    def __repr__(self):
        return f"DataAttribute class of '{'location' if self.location is not None else 'name'} {self.key}' layer"
    
    @property
    def key(self)->Union[int, str]:
        """Todo

        Returns
        -------
        Union[int, str]
            Todo
        """
        return self.location if self.location is not None else self.name
    
    @property
    def shape(self)->Union[None, Tuple]:
        """Union[None, Tuple]: Node array의 shape입니다. shape의 값은 반드시 tuple 이어야 합니다.

        Raises
        ----------
        ValueError
            shape의 value가 tuple이 아닐때 예외를 출력합니다
        
        Examples
        ----------
        k.shape = (1,3,640,640)
        k.shape = [1,3,640,640] #raise ValueError
        
        """
        return self._shape
    @shape.setter
    def shape(self, value:Tuple):
        if not isinstance(value, tuple):
            msg = f'{__class__}.shape의 값으로 tuple을 기대했지만 {type(value)}이 들어왔습니다.'
            raise ValueError(msg)
        self._shape = value
    
    @property
    def dtype(self)->Union[None, str]:
        """Union[None, str]: Node에 저장된 raw data의 type입니다.

        float32, float64, int32, uint32... 의 값을 가질 수 있습니다.

        Examples
        ----------
        k.dtype = 'float32'
        
        """
        return self._dtype
    @dtype.setter
    def dtype(self, value):
        self._dtype = EnumNodeRawDataType(value).value
    
    @property
    def format(self)->Union[None, str]:
        """Union[None, str]: Node의 shape이 어떤 포맷인지 저장합니다.

        포맷으로 nchw, nwhc, ncwh, nhwc... 의 값을 가질 수 있습니다.

        Examples
        ----------
        k.format = 'nchw'
        
        """
        return self._format
    @format.setter
    def format(self, value):
        self._format = EnumInputNodeShapeFormat(value).value
    
    @property
    def location(self)->Union[None, int]:
        """Union[None, int]: Node의 위치"""
        return self._location
    @location.setter
    def location(self, value):
        self._location = value
    
    @property
    def name(self)->Union[None, str]:
        """Union[None, str]: Node의 이름"""
        return self._name
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def height(self)->Union[None, int]:
        """Union[None, int]: Node에서 높이에 해당하는 값을 출력합니다."""
        if self._height is None:
            if (self._shape and self._format) is None:
                return None
            if len(self._format) != len(self._shape):
                return None
            self._height = dict(zip(self._format, self._shape)).get("h")
        return self._height

    @property
    def width(self)->Union[None, int]:
        """Union[None, int]: Node에서 넓이에 해당하는 값을 출력합니다."""
        if self._width is None:
            if (self._shape and self._format) is None:
                return None
            if len(self._format) != len(self._shape):
                return None
            self._width = dict(zip(self._format, self._shape)).get("w")
        return self._width
    
    @classmethod
    def get_props(cls)->List:
        """class의 property들을 list 타입으로 출력합니다"""
        return [x for x in dir(cls)
                if isinstance(getattr(cls, x), property)]
