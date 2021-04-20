from functools import partial
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import numpy as np
from scipy.spatial.transform import Rotation as R

import scipy.stats as stats
import sys
import getopt
import argparse


old_init = RigidRegistration.__init__
def new_init(self, *k, springs = None, initialization = True,
             prev_transform= None, next_transform = None, iter = 0, **kw ):
    old_init(self, *k, **kw)
    self.alpha = 2**5
    self.initialization = initialization
    self.springs = springs
    self.iter = iter
    self.max_iterations = 50
    #attach the springs to the respective vertebra
    #and extend the probability matrix.
    if initialization == False:
        for i, pt in enumerate(self.springs[self.iter]):

            self.X = np.vstack([self.X, pt])

            P_t = np.zeros_like(self.P[:, 0])
            P_t[-1] = 2 * self.sigma2 * self.alpha

            self.P = np.column_stack((self.P, P_t))



RigidRegistration.__init__ = new_init
def update_transform(self):
    """
    Calculate a new estimate of the rigid transformation.
    """


    if self.initialization:
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
    else:
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0),
                        self.Np + 2 * self.sigma2 * self.alpha)
        # source point cloud mean
        muY = np.divide(
            np.sum(np.dot(np.transpose(self.P), self.Y), axis=0),
            self.Np + 2 * self.sigma2 * self.alpha)

        self.X_hat = self.X - np.tile(muX, (self.N+self.springs[self.iter].shape[0], 1))
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
    # self.s = np.trace(np.dot(np.transpose(self.A),
    #                 np.transpose(self.R))) / self.YPY
    #size is fixed no need to rescale
    self.s = 1.0
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
             (2 * self.sigma2) + self.D * self.Np / 2 * np.log(self.sigma2)

    self.diff = np.abs(self.q - qprev)
    self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
    if self.sigma2 <= 0:
        self.sigma2 = self.tolerance / 10


RigidRegistration.update_variance = update_variance #monkey patch
RigidRegistration.update_transform = update_transform

#iterative visualization of the registration process
def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', alpha=0.1)
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', alpha=0.1)
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    # ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main(X_paths, Y_paths, sp, output_path, X_init_path=None):

    #load the point clouds
    X_list = []
    for p in X_paths:
        X_list.append(np.loadtxt(p))
    # X_list.append(np.loadtxt('./models/L2_raycasted_pc.txt')[::5])
    # X_list.append(np.loadtxt('./models/L3_raycasted_pc.txt')[::5])
    # X_list.append(np.loadtxt('./models/L4_raycasted_pc.txt')[::5])
    # X_list.append(np.loadtxt('./models/L5_raycasted_pc.txt')[::5])
    Y = np.loadtxt(Y_paths)
    # #load the springs
    sp = np.loadtxt(sp)[:,:3]
    springs = np.split(sp, 5)[::-1]

    output_initial = []
    output_list = []

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    callback = partial(visualize, ax=ax)
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([0, 200])

    if X_init_path != None:
    #run cpd without constraints (initial registration)
        print("entered init method: ")
        X_init = np.loadtxt(X_init_path)
        reg = RigidRegistration(**{'Y': X_init[::15, :3], 'X': Y[::200, :3],
                                   "initialization": True})
        TY, (s_reg, R_reg, t_reg) = reg.register(callback)

        np.savetxt(output_path, TY)

    else:
        #run CPD with constraints iteratively for each vertebra
        for i, X in enumerate(X_list):
            reg = RigidRegistration(**{'X': X[::15, :3], 'Y': Y[::200, :3],
                                       'springs': springs,
                                       "initialization": False, "iter":i})
            TY, (s_reg, R_reg, t_reg) = reg.register(callback)
            TY_ = np.dot(X[:,:3], np.linalg.inv(R_reg[:3,:3]))
            TY_[:,:3] += t_reg
            output_list.extend(TY_)
            print("iteration: ", i)

        np.savetxt(output_path, np.array(output_list))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-X_init", "--CT")
    parser.add_argument("-X", "--vertebrae", required=True, nargs='+', default=[])
    parser.add_argument("-Y", "--US", required=True)
    parser.add_argument("-s", "--springs", required=True)
    parser.add_argument("-save", "--saveas", required=True)
    # parser.add_argument('-method', '--method', required=True)
    args = parser.parse_args()
    main(args.vertebrae, args.US, args.springs, args.saveas, X_init_path=args.CT)