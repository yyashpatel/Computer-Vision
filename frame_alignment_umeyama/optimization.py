#!/usr/bin/env python3
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
from scipy.spatial.transform import Rotation as R
import collections
import pytransform3d.visualizer as pv
import struct
import transformations as tfs
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 10
# plt.switch_backend('agg')



# vicon poses
c2w_vio = np.load('/home/yash/catkin_ws/box_vicon.npy')

# colmap poses (world to camera)
c2w_colmap= np.load('/home/yash/box_poses.npy')

model = c2w_vio[...,3]
model = np.delete(model, 3, 1)
data = c2w_colmap[...,3]



def get_best_yaw(C):
    '''
    maximize trace(Rz(theta) * C)
    '''
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta


def rot_z(theta):
    R = tfs.rotation_matrix(theta, [0, 0, 1])
    R = R[0:3, 0:3]

    return R

def align_umeyama(model, data, known_scale=False, yaw_only=False):

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    return s, R, t


#print(model, data)
scale, rotation, translation = align_umeyama(data, model)
print(scale, rotation,translation)
#s,R,t = align_umeyama(model,data)

x_ho = []
y_ho = []
z_ho = []

x_h = []
y_h = []
z_h = []

x_ho_map = []
y_ho_map = []
z_ho_map = []

col_dist = []
for i in range(len(c2w_vio)):

    pose = c2w_vio[i] 
    t= c2w_vio[i][:3,3]
    #pose = np.delete(pose,3,0)
    x_ho.append(t[0])
    y_ho.append(t[1])
    z_ho.append(t[2])

    #print(pose.shape, rotation.shape, translation.shape) 
    #aligned translation
    aligned = scale*rotation.dot(pose[:3,3])+translation

    # #aligned rotation
    rot = rotation @ pose[:3,:3]
    #aligned T
    poses =  np.vstack([np.hstack([rot, np.array([[aligned[0]], [aligned[1]], [aligned[2]]])]), np.array([0., 0., 0., 1.]).reshape((1, 4))])
    #col_dist.append(poses)
    q = poses[:3,3]
    
    x_ho_map.append(q[0])
    y_ho_map.append(q[1])
    z_ho_map.append(q[2])

    orig = c2w_colmap[i]

    x_h.append(orig[:3,3][0])
    y_h.append(orig[:3,3][1])
    z_h.append(orig[:3,3][2])


fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# plotting
# ax.plot3D(x_ho, y_ho, z_ho, color = "blue",label = 'hall_object_orig', linewidth = 2)
# ax.plot3D(x_ho_map, y_ho_map, z_ho_map, color = "green", label = 'hall_object_mapped',linewidth = 2)
# ax.plot3D(x_h, y_h, z_h, color = "black", label = 'hall', linewidth = 2)
# ax.legend()
# ax.plot3D(x_t, y_t, z_t, color = "orange")

ax.scatter3D(x_ho, y_ho, z_ho, color = "blue")
ax.scatter3D(x_ho_map, y_ho_map, z_ho_map, color = "green")
ax.scatter3D(x_h, y_h, z_h, color = "black")
# ax.scatter3D(x_t, y_t, z_t, color = "black")

ax.set_title('camera poses COLMAP')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
