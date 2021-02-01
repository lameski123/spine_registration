import os
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt



os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ';C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;'

import imfusion
imfusion.init()

##############RAYCAST##################
# from myAlg import *
# from Raycast import *
# imgs = imfusion.open("C:\\Users\\Jane\\Downloads\\vertebra_imf_file")
# temp = imgs[0][0]
# # print(temp)
# output = imfusion.executeAlgorithm("Raycast", imgs)
# img = output

# print(img)
# # images = []
# # for i in imgs:
# #     images.append(np.array(i))
# # images
#
# img = np.squeeze(np.array(img[0]))
#
#
# #show raycast from all slices one by one
# for k in range(img.shape[0]):
#     # rays[k] = raycast(blur(img[k, :, :]), rays[k])
#     plt.figure()
#     plt.imshow(img[k])
#     # ax[1].imshow(blur(img[k, :, :]))
#     plt.show()
#
##############CPD##################


from CPD import *
from Raycast import *

XY = imfusion.SharedImageSet()
image = imfusion.open("vert_lumb_joined.imf")

# x_np = np.squeeze(np.array(X[0]))
# y_np = np.squeeze(np.array(Y[0]))
# # print(x_np.shape, y_np.shape)
# y_np_temp = np.zeros_like(x_np)
# # print(y_np_temp.shape)
# y_np_temp[0:52, 0:90, 0:102] = y_np
# # print(x_np.shape, y_np.shape)
# X = imfusion.SharedImage(np.expand_dims(x_np, axis=-1))
# Y = imfusion.SharedImage(np.expand_dims(y_np_temp, axis=-1))
# print(X[0])
# X.add(Y)
# XY.add(X)
# XY.add(Y)
print(image)
output = imfusion.executeAlgorithm("CPD", image)
#
print(output)
# Y = imfusion.executeAlgorithm('Apply Transformation', X)

src = np.squeeze(np.array(image[0][0]))
ray = np.squeeze(np.array(output[0]))
print(src.shape, ray.shape)

for k in range(src.shape[0]):
    # rays[k] = raycast(blur(img[k, :, :]), rays[k])
    if np.unique(src[k, :, :]).size > 0:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(src[k, :, :])
        ax[1].imshow(ray[k, :, :])
        plt.show()
#     # ax[1].imshow(blur(img[k, :, :]))
# # plt.show()

