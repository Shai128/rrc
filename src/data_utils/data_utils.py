from enum import Enum
from enum import auto


class DataType(Enum):
    Real = auto()
    Synthetic = auto()

class DataPart(Enum):
    Train = auto()
    Validation = auto()
    Test = auto()



