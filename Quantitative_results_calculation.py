from scipy.spatial.transform import Rotation as R
import numpy as np

rot_mat = [[ 0.84287787, -0.09003987, 0.53051835],
 [ 0.15993305, 0.98326722, -0.08721805],
 [-0.5137882, 0.15836158, 0.84317453]]
tr_mat = [ -3.04650206, -17.62546796, 0.94398422]

ct_res_rot = [-6.51, -32.28, -5.96]
ct_res_tr = [-3.21, -18.01, 0.64]
ct_rot_ = R.from_rotvec(np.array(ct_res_rot))
ct_rot_mat = ct_rot_.as_matrix()
ct_transform = np.eye(4)
ct_transform[:3,:3] = ct_rot_mat
ct_transform[:3,3] = ct_res_tr
print("CT-Transformation matrix: \n", ct_transform)

label_rot = [-2.17, -31.28, -7.78]
label_tr = [-15.21, -13.02, -1.53]
label_rot_ = R.from_rotvec(np.array(label_rot))
label_rot_mat = label_rot_.as_matrix()
label_transform = np.eye(4)
label_transform[:3,:3] = label_rot_mat
label_transform[:3,3] = label_tr
inv_label_transform = np.linalg.inv(label_transform)
print("Label-Transformation matrix: \n",label_transform)
print("Invers_Label-Transformation matrix: \n",inv_label_transform)
print("####QUATERNIONS####")
print(ct_rot_.as_quat())
print(label_rot_.as_quat())
print("####L2-Distance####")
print("Matrices: ", np.linalg.norm(ct_transform - inv_label_transform))
print("Quats: ", np.linalg.norm(ct_rot_.as_quat() - label_rot_.as_quat()))