from basic_transforms import *

class TrainAugmentation():
    def __init__(self, image_size, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops

        self.augment = Compose([RandomScale(scales=[0.5, 1, 2]),
                                Pad(size=image_size),
                                RandomCrop(size=image_size), 
                                RandomFlip(prob=0.5), 
                                ConvertDataType(), 
                                Normalize(mean_val, std_val)])
        
    def __call__(self, image, label):
        return self.augment(image, label)


class ValAugmentation():
    def __init__(self, image_size, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops

        self.augment = Compose([Resize(size=image_size),
                                ConvertDataType(), 
                                Normalize(mean_val, std_val)])
        
    def __call__(self, image, label):
        return self.augment(image, label)
