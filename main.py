import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt



os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ';C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;'

import imfusion
imfusion.init()

from myAlg import *
from Raycast import *
imgs = imfusion.open("C:\\Users\\Jane\\Downloads\\vertebra_imf_file")
# print(imgs)
output = imfusion.executeAlgorithm("Raycast", imgs)
img = output

print(img)
# images = []
# for i in imgs:
#     images.append(np.array(i))
# images

img = np.squeeze(np.array(img[0]))


#show raycast from all slices one by one
for k in range(img.shape[0]):
    # rays[k] = raycast(blur(img[k, :, :]), rays[k])
    plt.figure()
    plt.imshow(img[k])
    # ax[1].imshow(blur(img[k, :, :]))
    plt.show()

