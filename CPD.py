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
    self.s = 1
    # np.trace(np.dot(np.transpose(self.A),
                    # np.transpose(self.R))) / self.YPY
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
    def __init__(self, us, ct):
        super().__init__()
        # self.X = data[0]
        # self.Y = data[1]
        self.us = us
        self.ct = ct
        self.imageset_out = imfusion.SharedImageSet()

    @classmethod
    def convert_input(cls, data):
        return data

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        # rigid registration
        #remove dimensions with value 1
        # x_arr = np.squeeze(np.array(self.imageset1))# creates a copy
        # y_arr = np.squeeze(np.array(self.imageset2))
        #
        # #take the positions where a point exists
        # X = np.array(np.nonzero(x_arr)).T
        # Y = np.array(np.nonzero(y_arr)).T
        X = np.array(self.us)
        Y = np.array(self.ct)

        print(X, Y)
        #add an extra value 1 for fourth dimension since matrix = (4x4)
        # X_temp = np.ones((X.shape[0], X.shape[1] + 1))
        # Y_temp = np.ones((Y.shape[0], Y.shape[1] + 1))

        # for i in range(X.shape[0]):
        #     X_temp[i, :3] = np.multiply(X[i], self.imageset1.spacing)#@self.imageset[0].matrix@self.imageset[0].spacing
        # for i in range(Y.shape[0]):
        #     Y_temp[i, :3] = np.multiply(Y[i], self.imageset2.spacing)
        # x_matrix = self.imageset1.matrix
        # y_matrix = self.imageset2.matrix
        #align to the world position
        # print(x_matrix, y_matrix)
        #both y_matrix small missalignment
        #both x rotated and missalignment
        # X = X_temp@x_matrix
        # Y = Y_temp@y_matrix
        # X_ = np.zeros(X[0::200, :3].shape)
        # 0 1 2---
        # 0 2 1---
        # 1 2 0---
        # 1 0 2---
        # 2 0 1---
        # 2 1 0 winner (still translated probably center of image alignment problem)
        # X_[:, 0] = X[0::200, 2]
        # X_[:, 1] = X[0::200, 1]
        # X_[:, 2] = X[0::200, 0]

        # np.savetxt("inputX_point_cloud210.txt", X_)
        # np.savetxt("inputY_point_cloud.txt", Y[0::3,:3])
        reg = cpd(**{"X":X, "Y":Y})

        #make sure the scale is not a parameter that we care about
        TY, (s_reg, R_reg, t_reg) = reg.register()
        TY_map = np.zeros(max(TY[:,0])+5,max(TY[:,1])+5,max(TY[:,2])+5,)#\
        for i in TY:
            TY_map[int(i[0]), int(i[1]), int(i[2])] = 1
        # # np.savetxt("output_point_cloud.txt", TY)
        image_out = imfusion.SharedImage(np.expand_dims(TY_map, axis=-1))
        # #imgset2.spacing and imgset1.matrix small missalignment
        # #imageset2.spacing and imageset2.matrix 180 deg rotated and same small miissalingnment
        # #imageset1.spacing and imageset2.matrix 180 deg rotated and original is large
        # image_out.spacing = self.imageset2.spacing
        # image_out.matrix = self.imageset2.matrix
        self.imageset_out.add(image_out)
        print("file created!")
        np.savetxt("output_pc.txt", TY)

        return

    def output(self):
        return [self.imageset_out]


imfusion.registerAlgorithm('CPD', CPD)
