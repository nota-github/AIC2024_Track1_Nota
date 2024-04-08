class NotaRuntimeError(Exception):
    """Todo"""
    def __init__(self, msg=''):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f'[+] {self.msg}'

class AlreadyInitializedError(NotaRuntimeError):
    """Todo
    
    Arguments
    --------
    Class: cls

    """
    def __init__(self, cls=None):
        msg = f"이미 초기화 된 '{cls or 'class'}'입니다. finalize 함수 호출 후 다시 초기화 요청을 요청해주세요."
        super().__init__(msg)

class UninitializedError(NotaRuntimeError):
    """Todo"""
    def __init__(self,  cls=None):
        msg = f"초기화 되지 않은 '{cls or 'class'}'입니다. 초기화 후 인스턴스를 만들어주세요."
        super().__init__(msg)

class BasemodelError(NotaRuntimeError):
    """Todo"""

class UnsupportedFunction(NotaRuntimeError):
    """Todo"""

def unsupported_initialize():
    """Todo"""
    raise UnsupportedFunction("'initialize'은 'class' instance에서 호출이 불가능합니다.")

def unsupported_finalize():
    """Todo"""
    raise UnsupportedFunction("'finalize'은 'class' instance에서 호출이 불가능합니다.")

class InvalidPreprocessDataError(NotaRuntimeError):
    """Todo"""