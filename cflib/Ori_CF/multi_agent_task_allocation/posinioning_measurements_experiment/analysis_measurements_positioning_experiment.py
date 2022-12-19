import numpy as np
import matplotlib.pyplot as plt
import os
import random
from mpl_toolkits.mplot3d import Axes3D

fix_values_exp2 = np.array([0.062,0.051,0.035])
fix_values_exp1 = fix_values_exp2.copy()/2
error_2d = True
threshold = 0.1

# --------------------------------- Optimal configuration
pos_lps_arr = []
pos_vicon_arr = []
error_arr = []
sigma_arr = []
nodes = np.array([[1.00	,1.30	,0.23],[1.12	,1.40,	2.23],[1.06	,-1.14	,0.18],[1.13	,-1.32	,2.17],[-0.98	,-1.15	,0.16],[-0.84,	-1.28,	2.17],[-0.94	,1.36,	0.21],[-0.78	,1.44	,2.23]])
for i in range(1,82):
    measure_num = str(i)
    pos_lps = np.load('cflib/Ori_CF/multi_agent_task_allocation/posinioning_measurements_experiment/results/optimal_config/LPS_optimal/lps_static_pos_optimal_config_'+measure_num+'.npy')
    pos_vicon = np.load('cflib/Ori_CF/multi_agent_task_allocation/posinioning_measurements_experiment/results/optimal_config/vicon_optimal/vicon_optimal_config_'+measure_num+'.npy')
    pos_lps_avg = np.average(pos_lps, axis=0)


    error_ = np.abs(pos_vicon - pos_lps_avg)
    if error_2d:
        error = np.array([max(0, error_[0]-fix_values_exp2[0]),max(0, error_[1]-fix_values_exp2[1]), 0])
    else:
        error = np.array([max(0, error_[0]-fix_values_exp2[0]),max(0, error_[1]-fix_values_exp2[1]),max(0, error_[2]-fix_values_exp2[2])])
           
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



pos_lps_arr = np.array(pos_lps_arr)
pos_vicon_arr = np.array(pos_vicon_arr)
error_arr = np.array(error_arr)
sigma_arr = np.array(sigma_arr)


factor = error_arr  / threshold #smaller- green
colors = []
for fac in factor:
    colors.append([fac, 1-fac,0])
colors = np.array(colors)

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pos_vicon_arr[:,0], pos_vicon_arr[:,1], pos_vicon_arr[:,2],c=colors)
ax.scatter(nodes[:,0],nodes[:,1],nodes[:,2],c='blue', s=100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('color mapping error-optimal_comfiguration')

# histogram
fig2 = plt.figure(2)
ax2 = fig2.add_subplot('111')
ax2.hist(error_arr)
ax2.set_xlabel('error [meter]')
ax2.set_title('error histogram-optimal_comfiguration')
# plt.show()



# --------------------------------- Extended configuration

pos_lps_arr = []
pos_vicon_arr = []
error_arr = []
sigma_arr = []
nodes = np.array([[1.00	,1.30	,0.23],[1.12	,1.40,	2.23],[1.06	,-1.14	,0.18],[1.13	,-1.32	,2.17],[-1,	-1.14	,0.12],[-2.31	,-1.30	,2.17],[-0.96,	1.38	,0.18],[-2.33	,1.45	,2.22]])
for i in range(1,88):
    measure_num = str(i)
    pos_lps = np.load('cflib/Ori_CF/multi_agent_task_allocation/posinioning_measurements_experiment/results/extended_config/LPS_extended/lps_static_pos_changed_config_'+measure_num+'.npy')
    pos_vicon = np.load('cflib/Ori_CF/multi_agent_task_allocation/posinioning_measurements_experiment/results/extended_config/vicon_extended/vicon_changed_config_'+measure_num+'.npy')
    pos_lps_avg = np.average(pos_lps, axis=0)

    error_ = np.abs(pos_vicon - pos_lps_avg)
    if error_2d:
        error = np.array([max(0, error_[0]-fix_values_exp2[0]),max(0, error_[1]-fix_values_exp2[1]), 0])
    else:
        error = np.array([max(0, error_[0]-fix_values_exp2[0]),max(0, error_[1]-fix_values_exp2[1]),max(0, error_[2]-fix_values_exp2[2])])
           
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

pos_lps_arr = np.array(pos_lps_arr)
pos_vicon_arr = np.array(pos_vicon_arr)
error_arr = np.array(error_arr)
sigma_arr = np.array(sigma_arr)

factor = error_arr / threshold  #smaller- green
colors = []
for fac in factor:
    colors.append([fac, 1-fac,0])
colors = np.array(colors)

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')

ax3.scatter(pos_vicon_arr[:,0], pos_vicon_arr[:,1], pos_vicon_arr[:,2],c=colors)
ax3.scatter(nodes[:,0],nodes[:,1],nodes[:,2],c='blue', s=100)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_title('color mapping error-extended_comfiguration')

# histogram
fig4 = plt.figure(4)
ax4 = fig4.add_subplot('111')
ax4.hist(error_arr)
ax4.set_xlabel('error [meter]')
ax4.set_title('error histogram-extended_comfiguration')
plt.show()




# print(random.normalvariate(pos_lps_avg, sigma))
