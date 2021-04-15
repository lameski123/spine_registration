from functools import partial
import matplotlib.pyplot as plt
from pycpd import RigidRegistration
import numpy as np
from scipy.spatial.transform import Rotation as R

import scipy.stats as stats

old_init = RigidRegistration.__init__
def new_init(self, *k, springs = None, initialization = True,
             prev_transform= None, next_transform = None, iter = 0, **kw ):
    old_init(self, *k, **kw)
    self.alpha = 2**5
    self.initialization = initialization
    self.springs = springs
    self.iter = iter
    self.prev_transform = prev_transform
    self.next_transform = next_transform

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
        (s_prev, R_prev, t_prev) = (None, None, None)
        (s_next, R_next, t_next) = (None, None, None)
        reg_term = np.zeros((3,))
        if self.iter>0 and self.iter <3:
            (s_prev,R_prev,t_prev) = self.prev_transform
            (s_next,R_next,t_next) = self.next_transform
            reg_term = np.divide(
                self.alpha*s_prev*np.sum(np.dot(R_prev,np.transpose(self.springs[self.iter-1])), axis=0) +
                self.alpha*s_next*np.sum(np.dot(R_next,np.transpose(self.springs[self.iter+1])), axis=0),
                self.Np + 2*3*self.alpha)
        elif self.iter == 0:
            (s_prev, R_prev, t_prev) = self.prev_transform
            (s_next, R_next, t_next) = self.next_transform
            reg_term = np.divide(
                # self.alpha * s_prev * np.sum(np.dot(R_prev, np.transpose(self.springs[self.iter - 1])), axis=0) +
                self.alpha * s_next * np.sum(np.dot(R_next, np.transpose(self.springs[self.iter + 1])), axis=0),
                self.Np + 2 * 3 * self.alpha)
        elif self.iter == 3:
            (s_prev, R_prev, t_prev) = self.prev_transform
            (s_next, R_next, t_next) = self.next_transform
            reg_term = np.divide(
                self.alpha * s_prev * np.sum(np.dot(R_prev, np.transpose(self.springs[self.iter - 1])), axis=0) +
                self.alpha * s_next * np.sum(np.dot(R_next, np.transpose(self.springs[self.iter])), axis=0),
                self.Np + 2 * 3 * self.alpha)
        elif self.iter == 4:
            (s_prev, R_prev, t_prev) = self.prev_transform
            (s_next, R_next, t_next) = self.next_transform
            reg_term = np.divide(
                self.alpha * s_prev * np.sum(np.dot(R_prev, np.transpose(self.springs[self.iter - 1])), axis=0),
                # self.alpha * s_next * np.sum(np.dot(R_next, np.transpose(self.springs[self.iter + 1])), axis=0),
                self.Np + 2 * 3 * self.alpha)
        # target point cloud mean
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0),
                        self.Np + 2*3*self.alpha) - reg_term/2
        # source point cloud mean
        if self.iter<4:
            muY = np.divide(
                    np.sum(np.dot(np.transpose(self.P), self.Y), axis=0) +
                    2*self.alpha*np.sum(np.transpose(self.springs[self.iter]), axis=0),
                    self.Np + 2*3*self.alpha) - reg_term/2
        else:
            muY = np.divide(
                np.sum(np.dot(np.transpose(self.P), self.Y), axis=0) +
                2 * self.alpha * np.sum(np.transpose(self.springs[self.iter-1]), axis=0),
                self.Np + 2 * 3 * self.alpha) - reg_term / 2

        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        # centered source point cloud
        Y_hat = self.Y - np.tile(muY, (self.M, 1))
        self.YPY = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(Y_hat, Y_hat), axis=1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

    # print(self.A)
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



    # points = np.loadtxt('manualy_selected_pts (copy).txt')
    X_list = []
    X_list.append(np.loadtxt('./models/L1_pc.txt')[::5])
    X_list.append(np.loadtxt('./models/L2_pc_rt.txt')[::5])
    X_list.append(np.loadtxt('./models/L3_pc_rt.txt')[::5])
    X_list.append(np.loadtxt('./models/L4_pc_rt.txt')[::5])
    X_list.append(np.loadtxt('./models/L5_pc_rt.txt')[::5])
    # # synthetic data, equaivalent to X + 1
    Y = np.loadtxt('./models/L1_L5_pc.txt')
    sp = np.loadtxt('./models/manual_mid_points.txt')[:,:3]
    springs = np.split(sp, 4)
    # # _,_,X = grid_sampling(X)
    # print(X.shape)
    # print(Y.shape)
    # # print("Min-Y_X: ", min(Y[:,0]), " Max-Y_X: ", max(Y[:,0]))
    # # print("Min-Y_Y: ", min(Y[:,1]), " Max-Y_Y: ", max(Y[:,1]))
    # # print("Min-Y_Z: ", min(Y[:,2]), " Max-Y_Z: ", max(Y[:,2]))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # ax.scatter(points[:,0], points[:,1], points[:,2], color="red")
    # ax.set_xlim([-40,40])
    # ax.set_ylim([-40, 40])
    # ax.set_zlim([0, 200])
    #
    # line = np.zeros((200,3))
    # line[:,0] = 200*[0]
    #
    # y_axis_line_space = np.linspace(30,0,200).tolist()
    # line[:, 1] = 30-0.04*np.linspace(0,30,200)**2
    # line[:, 2] = range(200)
    # possible_points = []
    #
    # for i, coord in enumerate(X[:,:3]):
    #     for pt in line:
    #         if sum((coord - pt)**2) < 1:
    #
    #             possible_points.append(i)
    #
    # # print(possible_points)
    # possible_points = np.unique(possible_points)
    # np.savetxt("manualy_selected_pts.txt", X[possible_points])
    # ax.scatter(X[::50,0],X[::50,1],X[::50,2])
    # print(X[possible_points[0]])
    # plt.plot(X[possible_points,:3])
    # ax.scatter(X[possible_points,0],X[possible_points,1],X[possible_points,2], color="red")
    # plt.plot(line[:,0],line[:,1],line[:,2], "g+")

    # callback = partial(visualize, ax=ax)

    TY_list =[]
    X_ = np.array([x for x_x in X_list for x in x_x])
    reg = RigidRegistration(**{'X': X_[::5,:3], 'Y': Y[0::20,:3]})
    TY, (s_reg, R_reg, t_reg) = reg.register()
    # (s_reg, R_reg, t_reg) = (1,np.array([[0.99894673, 0.01998482, 0.04130427],
    #  [-0.02140603, 0.99918378, 0.0342572],
    #  [-0.04058593, -0.03510528, 0.99855916]]), np.array([8.94688965, 3.25655959, 2.46492813]))
    (s_reg_next, R_reg_next, t_reg_next) = (s_reg, R_reg, t_reg)
    for X in X_list:
        TY_ = np.dot(X[:, :3], R_reg[:3, :3])
        TY_[:, :3] += t_reg
        TY_list.append(TY_)
    output_list = []
    print("fiirst cpd passed!")
    for i, X in enumerate(TY_list):
        reg = RigidRegistration(**{'X': X[::5, :3], 'Y': Y[0::20, :3],
                                   'springs': springs, "prev_transform": (s_reg, R_reg, t_reg),
                                   'next_transform':(s_reg_next, R_reg_next, t_reg_next),
                                   "initialization": False, "iter":i})
        TY, (s_reg, R_reg, t_reg) = reg.register()
        TY_ = np.dot(X[::5,:3], R_reg[:3,:3])
        TY_[:,:3] += t_reg
        output_list.extend(TY_)
        print("iteration: ", i)
    # print(R_reg, t_reg)
    np.savetxt("./models/bio_experiment_one_by_one2.txt", np.array(output_list))


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.scatter(points[:,0], points[:,1], points[:,2], color="red")
    ax.set_xlim([-40,40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([0, 200])
    ax.scatter(np.array(output_list)[::5, 0], np.array(output_list)[::5, 1], np.array(output_list)[::5, 2], color="red")
    ax.scatter(Y[::20,0], Y[::20,1], Y[::20,2], color = "blue")
    plt.show()
    # rot_mat = [[0.87228969, -0.15346405, 0.46428384],
    #            [0.16356022, 0.98635545, 0.01873468],
    #            [-0.460824, 0.0595963, 0.8854883]]
    # tr_mat = [-9.40348396, -9.78221285, -1.18731904]
    #
    # rotR = R.from_matrix(rot_mat)
    # rot_vec = rotR.as_euler('xyz', degrees=True)
    # cpdR_transform = np.eye(4)
    # cpdR_transform[:3, :3] = rot_mat
    # cpdR_transform[:3, 3] = tr_mat
    # ct_res_rot = [-3.08, -30.68, -6.31]
    # ct_res_tr = [1.05, -15.21, -0.11]
    # ct_rot_ = R.from_euler("xyz", np.array(ct_res_rot), degrees=True)
    # ct_rot_mat = ct_rot_.as_matrix()
    # ct_transform = np.eye(4)
    # ct_transform[:3, :3] = ct_rot_mat
    # ct_transform[:3, 3] = ct_res_tr
    # inverse_ct = np.linalg.inv(ct_transform)
    #
    # TY = np.dot(Y[0::5,:3], inverse_ct[:3,:3])
    # TY[:,:3] += ct_res_tr
    # np.savetxt("./models/testpc_out_ground_truth_test.txt", TY)
    # plt.show()


if __name__ == '__main__':
    main()