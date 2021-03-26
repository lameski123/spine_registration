'''
https://github.com/mariatirindelli/VertebraRegistration/blob/develop/ImFusionPlugins/ReplaceImagesInStream.py
'''

import imfusion
import numpy as np
import json
import os


class ReplaceImagesInStream(imfusion.Algorithm):
    def __init__(self, tracked_stream, untracked_stream):
        super().__init__()
        self.tracked_stream = tracked_stream
        self.untracked_stream = untracked_stream
        self.imageset_out = imfusion.SharedImageSet()

    @classmethod
    def convert_input(cls, data):

        if len(data) != 2:
            raise imfusion.IncompatibleError('Requires two inputs')

        if not isinstance(data[0], imfusion.SharedImageSet) or data[0].modality != imfusion.Data.Modality.ULTRASOUND:
            raise imfusion.IncompatibleError('First input must be a 3D Sweep')

        if not isinstance(data[1], imfusion.SharedImageSet) or data[1].modality != imfusion.Data.Modality.LABEL:
            raise imfusion.IncompatibleError('Second input must be a 2D label map')

        return data

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        for i, img in enumerate(self.tracked_stream):
            label_array = np.array(self.untracked_stream[i])
            label_array[label_array>0] = 100

            arr = np.array(img, copy=False)
            arr[:, :, :] = label_array
            img.setDirtyMem()

        return


imfusion.registerAlgorithm('Replace Images In Stream', ReplaceImagesInStream)