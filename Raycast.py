import imfusion
import numpy as np


def raycast(image, rays):

    for i in range(image.shape[1]-1, 0, -1):
        for j in range(image.shape[0]-1, 0, -1):
            if image[j, i] != 0:
                rays[j, i] = 1
                break
    return rays


def blur(a):
    kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

class RayCast(imfusion.Algorithm):
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
            # print(image.size())
            arr = np.array(image) # creates a copy
            rays = np.zeros_like(np.squeeze(arr)) #empty image
            # print(arr.shape, rays.shape)
            for k in range(arr.shape[0]):
                rays[k] = raycast(blur(arr[k, :, :]), rays[k]) #assign the rays

            # create the output image from the thresholded data
            # print(np.expand_dims(rays, axis=-1).shape)
            image_out = imfusion.SharedImage(np.expand_dims(rays, axis=-1))
            self.imageset_out.add(image_out)

    def output(self):
        return [self.imageset_out]


imfusion.registerAlgorithm('Raycast', RayCast)