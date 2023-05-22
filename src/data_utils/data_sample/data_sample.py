import abc


class DataSample(abc.ABC):
    def __init__(self, x, **kwargs):
        self.x = x


class LabeledDataSample(DataSample):
    def __init__(self, x, y, **kwargs):
        DataSample.__init__(self, x=x, **kwargs)
        self.y = y