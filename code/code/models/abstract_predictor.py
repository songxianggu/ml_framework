from abc import ABC, abstractmethod

class AbstractPredictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, features: []) -> float:
        pass
