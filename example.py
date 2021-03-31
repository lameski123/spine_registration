from functools import partial
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import numpy as np

import scipy.stats as stats

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


RigidRegistration.update_variance = update_variance #monkey patch
RigidRegistration.update_transform = update_transform

def grid_sampling(sample):
    intensity = sample[:,2]
    x = sample[:,0]
    y = sample[:,1]
    binx=np.linspace(-1000.,1000.,2000)
    biny=np.linspace(-1000.,1000.,2000)
    # binz=np.linspace(-10.,1000.,2000)
    # print(intensity)
    ret = stats.binned_statistic_2d(x,y, intensity, 'mean', bins=[binx,biny])

    Z=ret.statistic

    Z = np.ma.masked_invalid(Z) # allow to mask Nan values got in bins where there is no value

    binx_centers = (binx[1:] + binx[:-1]) / 2
    biny_centers = (biny[1:] + biny[:-1]) / 2
    # binz_centers = (binz[1:] + binz[:-1]) / 2

    Xcenters, Ycenters = np.meshgrid(binx_centers, biny_centers)
    xnew = np.ma.masked_array(Xcenters, Z.mask).compressed()
    ynew = np.ma.masked_array(Ycenters, Z.mask).compressed()
    znew = Z.compressed()

    return xnew,ynew,znew

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='US', alpha=0.1)
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='CT', alpha=0.1)
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    X = np.loadtxt('./models/testpc_us.txt')
    # synthetic data, equaivalent to X + 1
    Y = np.loadtxt('./models/testpc_ct.txt')
    # _,_,X = grid_sampling(X)
    print(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = RigidRegistration(**{'X': X[0::500,:3], 'Y': Y[0::5,:3]})
    TY, (s_reg, R_reg, t_reg) = reg.register(callback)
    print(R_reg, t_reg)
    # np.savetxt("./models/testpc_out.txt", TY)
    plt.show()


if __name__ == '__main__':
    main()