import imfusion
import numpy as np
from pycpd import RigidRegistration as cpd


def update_transform(self):
    """
    Calculate a new estimate of the rigid transformation.
    """

    # target point cloud mean
    muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0),
                    self.Np)
    # source point cloud mean
    muY = np.divide(
        np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

    self.X_hat = self.X - np.tile(muX, (self.N, 1))
    # centered source point cloud
    Y_hat = self.Y - np.tile(muY, (self.M, 1))
    self.YPY = np.dot(np.transpose(self.P1), np.sum(
        np.multiply(Y_hat, Y_hat), axis=1))

    self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
    self.A = np.dot(self.A, Y_hat)

    # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
    U, _, V = np.linalg.svd(self.A, full_matrices=True)
    C = np.ones((self.D,))
    C[self.D - 1] = np.linalg.det(np.dot(U, V))

    # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
    self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
    # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
    self.s = np.trace(np.dot(np.transpose(self.A),
                             np.transpose(self.R))) / self.YPY
    self.t = np.transpose(muX) - self.s * \
             np.dot(np.transpose(self.R), np.transpose(muY))




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
cpd.update_transform = update_transform

class IncompatibleError(Exception):
    def __init__(self, str):
        self.str = str
        super().__init__(self.str)


class CPD(imfusion.Algorithm):
    def __init__(self, data):
        super().__init__()
        # self.X = data[0]
        # self.Y = data[1]
        self.imageset = data
        self.imageset_out = imfusion.SharedImageSet()

    @classmethod
    def convert_input(cls, data):
        # print(len(data))
        # if data.size == 2 and isinstance(data[0], imfusion.SharedImageSet):
        return data
        # else:
        #     raise IncompatibleError("Requires exactly two images")

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        # rigid registration
        # for x,y in zip(self.X, self.Y):
        #add an extra value 1 for fourth dimension since matrix = (4x4)
        x_arr = np.squeeze(np.array(self.imageset[1]))# creates a copy
        y_arr = np.squeeze(np.array(self.imageset[0]))

        X = np.array(np.nonzero(x_arr)).T
        Y = np.array(np.nonzero(y_arr)).T
        X_temp = np.ones((X.shape[0], X.shape[1]+1))
        Y_temp = np.ones((Y.shape[0], Y.shape[1] + 1))

        # print(X_temp.shape)
        for i in range(X.shape[0]):
            X_temp[i, :3] = np.multiply(X[i], self.imageset[0].spacing)#@self.imageset[0].matrix@self.imageset[0].spacing
        for i in range(Y.shape[0]):
            Y_temp[i, :3] = np.multiply(Y[i], self.imageset[0].spacing)

        X = X_temp@self.imageset[0].matrix
        Y = Y_temp@self.imageset[0].matrix
        # print(X_temp, X_temp.shape)
        # X = x_arr.ravel()
        # x_arr = X[X!=0]
        # x_arr = x_arr[::2]
        # Y = y_arr.ravel()
        # y_arr = Y[Y!=0]
        # y_arr = y_arr[::2]

        # print(X.T, Y.T.shape, type(X))
        #error thrown below y_arr must be 2D???
        reg = cpd(**{"X":X[:,:3], "Y":Y[:,:3]})
        # print(reg)
        #make sure the scale is not a parameter that we care about
        TY, (s_reg, R_reg, t_reg) = reg.register()
        TY_map = np.zeros_like(y_arr)
        for i in TY:
            TY_map[int(round(i[0])), int(round(i[1])), int(round(i[2]))] = 1

        image_out = imfusion.SharedImage(np.expand_dims(TY_map, axis=-1))
        image_out.spacing = self.imageset[0].spacing
        image_out.matrix = self.imageset[0].matrix
        self.imageset_out.add(image_out)

    def output(self):
        return [self.imageset_out]


imfusion.registerAlgorithm('CPD', CPD)