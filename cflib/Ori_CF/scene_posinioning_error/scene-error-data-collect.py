# this function connects to several CF logging data and collect CF relative positions


import time
import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
import numpy as np

def activate_high_level_commander(scf):
    scf.cf.param.set_value('commander.enHighLevel', '1')

uri_1 = 'radio://0/80/2M/E7E7E7E7E1'
uri_2 = 'radio://0/80/2M/E7E7E7E7E2'
uri_3 = 'radio://0/80/2M/E7E7E7E7E3'
uris = {uri_1, uri_2, uri_3}
uri_l = [uri_1, uri_2, uri_3]

drone_pos = []
for uri in uris:
    drone_pos.append([])

# set ground truth distance between drones
d12gt = 0.5
d23gt = 0.55
d13gt = 0.6
gt_dist_vec = np.array([d12gt , d23gt, d13gt])
filename='data_collect'
data = np.array([[0,0,0,0]]) # initiate


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(activate_high_level_commander)
        swarm.reset_estimators()
        # scf = swarm._cfs['radio://0/80/2M/E7E7E7E7E7']
        for i in range(1000):
            swarm.get_estimated_positions()
            print(i)
            for link_idx in range(len(uri_l)):
                x,y,z = swarm._positions[uri_l[link_idx]]
                print(f'drone: {link_idx}, x ={x: .3f}, y ={y: .3f}, z ={z: .3f}')
                drone_pos[link_idx] = np.array([x,y,z])
              
            d12 = np.linalg.norm(drone_pos[1] - drone_pos[2], ord=2)
            d23 = np.linalg.norm(drone_pos[2] - drone_pos[3], ord=2)
            d13 = np.linalg.norm(drone_pos[1] - drone_pos[3], ord=2)
            dist_vec_measured = np.array([d12, d23, d13])
            avg_position = (drone_pos[0] + drone_pos[1] + drone_pos[2]) / 3 
            error = np.linalg.norm(gt_dist_vec - dist_vec_measured, ord=2)
            result = np.append(avg_position, error)
            data = np.append(data, [result], axis=0)
            time.sleep(1)

        data = np.delete(data, [0], axis=0) # delete initiated row
        np.save(filename, data)

