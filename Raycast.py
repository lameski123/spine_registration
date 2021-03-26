import imfusion
import numpy as np


def raycast(image, rays):
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image[j, i] != 0:
                rays[j, i] = 1
                break
    return rays

def multi_raycast(image, rays):
    uni = np.unique(image)
    # # print(uni)
    uni = np.delete(uni, 0)
    # the index in image.shape[] changes according to image positioninig. First check Imfusion then change indexes

    for u in uni:
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if image[j, i] == u:
                    rays[j, i] = u
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
    def __init__(self, data):
        super().__init__()
        self.imageset = data
        self.imageset_out = imfusion.SharedImageSet()
        # self.axis = data[1]

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
            arr = np.squeeze(np.array(image)) # creates a copy
            rays = np.zeros_like(np.squeeze(arr)) #empty image

            #the index in arr.shape[] and the dimension that we iterate on K
            #changes according to image positioninig. First check Imfusion then change indexes
            print(arr.shape)
            for k in range(arr.shape[0]):
                rays[k,:,:] = raycast(arr[k,:,:], rays[k,:,:]) #assign the rays

            #assign the spacing and the transform matrix
            image_out = imfusion.SharedImage(np.expand_dims(rays, axis=-1))
            image_out.spacing = image.spacing
            image_out.matrix = image.matrix
            self.imageset_out.add(image_out)

    def output(self):
        return [self.imageset_out]


imfusion.registerAlgorithm('Raycast', RayCast)