import numpy as np
import matplotlib.pyplot as plt
import os
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.interpolate import griddata, Rbf

# print(random.normalvariate(pos_lps_avg, sigma))

def load_results(lps_dir, vicon_dir, fix_values,samples_num ,is_2d, threshold, nodes):
    pos_lps_arr = []
    pos_vicon_arr = []
    error_arr = []
    sigma_arr = []
    for i in range(1,samples_num+1):
        measure_num = str(i)
        pos_lps = np.load(lps_dir+measure_num+'.npy')
        pos_vicon = np.load(vicon_dir+measure_num+'.npy')
        pos_lps_avg = np.average(pos_lps, axis=0)
        error_ = np.abs(pos_vicon - pos_lps_avg)
        if is_2d:
            error = np.array([max(0, error_[0]-fix_values[0]),max(0, error_[1]-fix_values[1]), 0])
        else:
            error = np.array([max(0, error_[0]-fix_values[0]),max(0, error_[1]-fix_values[1]),max(0, error_[2]-fix_values[2])])
        error_size = np.linalg.norm(error, ord=2)
        sigma = np.std(pos_lps, axis=0)
        if error_size < threshold: #filter out threshold 
            pos_lps_arr.append(pos_lps_avg)
            pos_vicon_arr.append(pos_vicon)
            error_arr.append(error_size)
            sigma_arr.append(sigma)
            # mirror results
            if pos_vicon[1] > 0.3:
                mirror_point = np.array([pos_vicon[0], -pos_vicon[1], pos_vicon[2]])
                pos_vicon_arr.append(mirror_point)
                error_arr.append(error_size)
                sigma_arr.append(sigma)
    # rotate and translate 
    rot_mat = np.array([[np.cos(np.pi), -np.sin(np.pi), 0 ],[np.sin(np.pi), np.cos(np.pi), 0],[0,0,1]])
    pos_vicon_arr = np.array(pos_vicon_arr) 
    translate_x = 1.13
    pos_vicon_arr = np.transpose(rot_mat.dot( np.transpose(pos_vicon_arr))) + np.array([translate_x, 0, 0])
    nodes = np.transpose(rot_mat.dot( np.transpose(nodes))) + np.array([translate_x, 0, 0])
    pos_lps_arr = np.array(pos_lps_arr)
    error_arr = np.array(error_arr)
    sigma_arr = np.array(sigma_arr)
    # limits
    # print(np.min(pos_vicon_arr,axis=0))
    # print(np.max(pos_vicon_arr,axis=0))
    return pos_lps_arr, pos_vicon_arr, error_arr, sigma_arr, nodes


def plot_sampled_data(error_arr, threshold, title, pos_vicon_arr, nodes):
    factor = error_arr  / threshold #smaller- green
    colors = []
    for fac in factor:
        colors.append([fac, 1-fac,0])
    colors = np.array(colors)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos_vicon_arr[:,0], pos_vicon_arr[:,1], pos_vicon_arr[:,2],c=colors)
    ax.scatter(nodes[:,0],nodes[:,1],nodes[:,2],c='blue', s=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title+' - sampeled data')


def exterpolate_3d(limits, resolution, pos_vicon_arr, error_arr):
    grid_x, grid_y, grid_z = np.mgrid[limits[0][0]:limits[0][1]:resolution, limits[1][0]:limits[1][1]:resolution, limits[2][0]:limits[2][1]:resolution ]
    # interpolated_error = griddata(pos_vicon_arr, error_arr, (grid_x, grid_y, grid_z ), method='linear')
    rbf4 = Rbf(pos_vicon_arr[:,0], pos_vicon_arr[:,1], pos_vicon_arr[:,2],error_arr, function="multiquadric", smooth=5)
    interpolated_error = rbf4(grid_x, grid_y, grid_z)
    return grid_x, grid_y, grid_z, interpolated_error    


def plot_exterpolate(interpolated_error, threshold, grid_x, grid_y, grid_z, nodes,title):
    factor = interpolated_error  / threshold 
    colors = []
    for xi in range(factor.shape[0]):
        for yi in range(factor.shape[1]):
            for zi in range(factor.shape[2]):
                fac = factor[xi,yi, zi]
                if fac > 1:
                    fac = 1
                if fac < 0:
                    fac = 0
                if 0 <= fac <=1:
                    colors.append(np.array([fac, 1-fac, 0,1]))
                else:
                    colors.append( np.array([1,1, 1,0])) 

    colors = np.array(colors)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_x, grid_y, grid_z,c=colors)
    ax.scatter(nodes[:,0],nodes[:,1],nodes[:,2],c='blue', s=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title + ' - interpolated data')


def plot_histogram(error_arr, title):
    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.hist(error_arr)
    ax.set_xlabel('error [meter]')
    ax.set_title(title)

#general params

error_2d = True
threshold = 0.2
limits = [[-0.32, 3.5], [-1.5,1.5], [0.2,2.2]]
resolution = 0.3

# box config params:
samples_num_exp1 = 81
fix_values_exp1 = np.array([0.062,0.051,0.035])/2
nodes1 = np.array([[1.00	,1.30	,0.23],[1.12	,1.40,	2.23],[1.06	,-1.14	,0.18],[1.13	,-1.32	,2.17],[-0.98	,-1.15	,0.16],[-0.84,	-1.28,	2.17],[-0.94	,1.36,	0.21],[-0.78	,1.44	,2.23]])
optimal_lps_dir = 'cflib/Ori_CF/multi_agent_task_allocation/posinioning_measurements_experiment/results/optimal_config/LPS_optimal/lps_static_pos_optimal_config_'
optimal_vicon_dir = 'cflib/Ori_CF/multi_agent_task_allocation/posinioning_measurements_experiment/results/optimal_config/vicon_optimal/vicon_optimal_config_'
title_exp1 = 'box configuration'

# extended params
samples_num_exp2 = 87
fix_values_exp2 = np.array([0.062,0.051,0.035])/2
nodes2 =  np.array([[1.00	,1.30	,0.23],[1.12	,1.40,	2.23],[1.06	,-1.14	,0.18],[1.13	,-1.32	,2.17],[-1,	-1.14	,0.12],[-2.31	,-1.30	,2.17],[-0.96,	1.38	,0.18],[-2.33	,1.45	,2.22]])
extended_lps_dir = 'cflib/Ori_CF/multi_agent_task_allocation/posinioning_measurements_experiment/results/extended_config/LPS_extended/lps_static_pos_changed_config_'
extended_vicon_dir =  'cflib/Ori_CF/multi_agent_task_allocation/posinioning_measurements_experiment/results/extended_config/vicon_extended/vicon_changed_config_'
title_exp2 = 'extended configuration'

# box config
pos_lps_arr, pos_vicon_arr, error_arr, sigma_arr, nodes1 = load_results(lps_dir=optimal_lps_dir, vicon_dir=optimal_vicon_dir, fix_values=fix_values_exp1,samples_num=samples_num_exp1 ,is_2d=error_2d, threshold=threshold, nodes=nodes1)
plot_sampled_data(error_arr, threshold, title_exp1, pos_vicon_arr, nodes1)
grid_x, grid_y, grid_z, interpolated_error = exterpolate_3d(limits, resolution, pos_vicon_arr, error_arr)
plot_exterpolate(interpolated_error, threshold, grid_x, grid_y, grid_z, nodes1, title_exp1)


# extended configuration
pos_lps_arr, pos_vicon_arr, error_arr, sigma_arr, nodes2 = load_results(lps_dir=extended_lps_dir, vicon_dir=extended_vicon_dir, fix_values=fix_values_exp2,samples_num=samples_num_exp2 ,is_2d=error_2d, threshold=threshold, nodes=nodes2)
plot_sampled_data(error_arr, threshold, title_exp2, pos_vicon_arr, nodes2)
grid_x, grid_y, grid_z, interpolated_error = exterpolate_3d(limits, resolution, pos_vicon_arr, error_arr)
plot_exterpolate(interpolated_error, threshold, grid_x, grid_y, grid_z, nodes2, title_exp2)

plt.show()