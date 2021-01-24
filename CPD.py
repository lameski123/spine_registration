import imfusion
import numpy as np
from pycpd import RigidRegistration as cpd


def update_variance(self):
    """
    Update the variance of the mixture model using the new estimate of the rigid transformation.
    See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.
    """
    qprev = self.q

    trAR = np.trace(np.dot(self.A, self.R))
    xPx = np.dot(np.transpose(self.Pt1), np.sum(
        np.multiply(self.X_hat, self.X_hat), axis=1))
    self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / \
             (2 * self.sigma2) + self.D * self.Np / 2 * np.log(self.sigma2) #add the regularization here
    self.diff = np.abs(self.q - qprev)
    self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
    if self.sigma2 <= 0:
        self.sigma2 = self.tolerance / 10


cpd.update_variance = update_variance #monkey patch


class IncompatibleError(Exception):
    def __init__(self, str):
        self.str = str
        super().__init__(self.str)


class CPD(imfusion.Algorithm):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
        # self.imageset = imageset
        self.imageset_out = imfusion.SharedImageSet()

    @classmethod
    def convert_input(cls, data):
        if len(data) == 2:
            X, Y = data
        else: raise IncompatibleError("Requires exactly two images")

        return X, Y

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        # rigid registration
        # for x,y in zip(self.X, self.Y):
        x_arr = np.array(self.imageset[0]) # creates a copy
        # print(np.unique(x_arr))
        # x_arr = x_arr[x_arr < 2]
        y_arr = np.array(self.imageset[1])
        # y_arr = y_arr[y_arr > 2]
        reg = cpd(**{'X': x_arr, 'Y': y_arr})
        image_out = imfusion.SharedImage(reg)
        self.imageset_out.add(image_out)

    def output(self):
        return [self.imageset_out]


imfusion.registerAlgorithm('CPD', CPD)