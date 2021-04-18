from scipy.spatial.transform import Rotation as R
import numpy as np
import math
# results from bigger point cloud obtained axial
# [[ 0.84871837 -0.10206251  0.51890305]
#  [ 0.14736316  0.98797923 -0.04670271]
#  [-0.50789884  0.11610464  0.85355637]] [-15.96317142 -15.05570723   0.36232684]

#CPD for US axial
print("CPD results from point cloud obtained axial w.r.t the spine: ")
rot_mat = [[ 0.84287787, -0.09003987, 0.53051835],
 [ 0.15993305, 0.98326722, -0.08721805],
 [-0.5137882, 0.15836158, 0.84317453]]
tr_mat = [ -3.04650206, -17.62546796, 0.94398422]
rot = R.from_matrix(rot_mat)
rot_vec = rot.as_euler('xyz', degrees=True)
cpd_transform = np.eye(4)
cpd_transform[:3,:3] = rot_mat
cpd_transform[:3,3] = tr_mat
print("CPD - Rotation Vector: ", rot_vec)
print("CPD - Translation Vector: ", tr_mat)
print("CPD - Transformation matrix: \n", cpd_transform)

#cpd for US radial
print("CPD results from point cloud obtained radial w.r.t the spine: ")
rot_mat = [[ 0.87228969, -0.15346405,  0.46428384],
 [ 0.16356022,  0.98635545,  0.01873468],
 [-0.460824,    0.0595963,   0.8854883 ]]
tr_mat = [-9.40348396, -9.78221285, -1.18731904]

rotR = R.from_matrix(rot_mat)
rot_vec = rotR.as_euler('xyz', degrees=True)
cpdR_transform = np.eye(4)
cpdR_transform[:3,:3] = rot_mat
cpdR_transform[:3,3] = tr_mat
print("CPD - Rotation Vector: ", rot_vec)
print("CPD - Translation Vector: ", tr_mat)
print("CPD - Transformation matrix: \n", cpdR_transform)
#

ct_res_rot = [0.37, -30.68, -9.6]
ct_res_tr = [-10.68, -12.84, -2.02]

#prev ground truth
# ct_res_rot = [-3.08, -30.68, -6.31]
# ct_res_tr = [1.05, -15.21, -0.11]

ct_rot_ = R.from_euler("xyz", np.array(ct_res_rot), degrees=True)
ct_rot_mat = ct_rot_.as_matrix()
ct_transform = np.eye(4)
ct_transform[:3,:3] = ct_rot_mat

inverse_ct = np.linalg.inv(ct_transform)
inverse_rot = inverse_ct[:3,:3]
inverse_ct[:3,3] = ct_res_tr
inverse_r = R.from_matrix(inverse_rot)
inverse_rotation = inverse_r.as_euler("xyz", degrees=True)

print("Ground Truth - Rotation Vector: ", inverse_rotation)
print("Ground Truth - Translation Vector: ", ct_res_tr)
print("Ground Truth - Transformation matrix: \n", inverse_ct)

print("####QUATERNIONS####")

print(rot.as_quat())
print(rotR.as_quat())
print(inverse_r.as_quat())

print("####L2-Distance####")
print("Matrices axial: ", np.linalg.norm(cpd_transform - inverse_ct))
print("Matrices radial: ", np.linalg.norm(cpdR_transform - inverse_ct))
print("Quats axial: ", np.linalg.norm(rot.as_quat() - inverse_r.as_quat()))
print("Quats radial: ", np.linalg.norm(rotR.as_quat() - inverse_r.as_quat()))