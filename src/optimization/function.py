import abc

class Function(abc.ABC):
    """
    Abstract class of function f(x)
    """
    def __init__(self, in_features):
        self._in_features = in_features

    @abc.abstractmethod
    def __call__(self):
        pass

    @abc.abstractmethod
    def get_gradient(self):
        pass

    @property
    def in_features(self):
        return self._in_features