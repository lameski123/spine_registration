import imfusion
import numpy as np


class MyAlgorithm(imfusion.Algorithm):
    def __init__(self, imageset):
        super().__init__()
        self.imageset = imageset
        self.imageset_out = imfusion.SharedImageSet()

    @classmethod
    def convert_input(cls, data):
        # if len(data) == 1 and isinstance(data[0], imfusion.SharedImageSet):
        #     return data
        # raise RuntimeError('Requires exactly one image')
        return data

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        # compute the thresholding on each individual image in the set
        for image in self.imageset:
            arr = np.array(image) # creates a copy
            arr[arr < 2500] = 0
            arr[arr >= 2500] = 1

            # create the output image from the thresholded data
            image_out = imfusion.SharedImage(arr)
            self.imageset_out.add(image_out)

    def output(self):
        return [self.imageset_out]


imfusion.registerAlgorithm('My Algorithm', MyAlgorithm)