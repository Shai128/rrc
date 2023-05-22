from src.data_utils.data_sample.data_sample import DataSample, LabeledDataSample


class I2IDataSample(DataSample):
    def __init__(self, x, valid_image_mask, center_pixel, **kwargs):
        super().__init__(x, **kwargs)
        self.valid_image_mask = valid_image_mask
        self.center_pixel = center_pixel


class LabeledI2IDataSample(LabeledDataSample, I2IDataSample):
    def __init__(self, x, y, valid_image_mask, center_pixel, **kwargs):
        LabeledDataSample.__init__(self, x, y, **kwargs)
        I2IDataSample.__init__(self, x, valid_image_mask, center_pixel)