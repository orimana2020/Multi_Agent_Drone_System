import time
import cflib.crtp
import numpy as np
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.Ori_CF.fly_CF.Trajectory import Generate_Trajectory, upload_trajectory

class CF_flight_manager(object):
    def __init__(self, uri_list):
        self.drone_num = len(uri_list)
        uris = set(uri_list)
        self.uri_dict = dict()
        for i  in range(len(uri_list)):
            self.uri_dict[i] = uri_list[i]
        cflib.crtp.init_drivers()
        factory = CachedCfFactory(rw_cache='./cache')
        self.swarm = Swarm(uris, factory=factory)
        self.swarm.open_links()
        self.swarm.parallel_safe(self.activate_high_level_commander)
        self.swarm.reset_estimators()
        self.open_threads = []
        self.goal = []
        for _ in range(len(uri_list)):
            self.open_threads.append([])
            self.goal.append([])

       
    def activate_high_level_commander(self, scf):
        scf.cf.param.set_value('commander.enHighLevel', '1')


    def take_off(self, scf):
        cf = scf.cf
        commander = cf.high_level_commander
        commander.takeoff(1.0, 2.0)
        time.sleep(3.0) 


    def _execute_trajectory(self, scf, waypoints): 
        cf = scf.cf
        commander = cf.high_level_commander 
        x, y, z = waypoints[0]
        print('start wp = ', waypoints[0])
        commander.go_to(x, y, z, yaw=0, duration_s=1)
        time.sleep(1)
        try:
            trajectory_id = 1
            traj = Generate_Trajectory(waypoints, velocity=1, plotting=0, force_zero_yaw=False, is_smoothen=True)
            traj_coef = traj.poly_coef
            duration = upload_trajectory(cf, trajectory_id ,traj_coef)
            commander.start_trajectory(trajectory_id, 1.0, False)
            time.sleep(duration)
        except:
            print('failed to execute trajectory')
    
    def execute_trajectory_mt(self, drone_idx, waypoints):# send trajectory with multi thread mode
        thread = self.swarm.trajectory_to_drone(self._execute_trajectory, self.uri_dict[drone_idx], waypoints)
        self.open_threads[drone_idx] = thread
        self.goal[drone_idx] = waypoints[-1]
    
    def get_position(self, drone_idx):
        scf = self.swarm._cfs[self.uri_dict[drone_idx]]
        self.swarm.get_single_cf_estimated_position(scf)

        
    def reached_goal(self, drone_idx):
        try:
            self.get_position(drone_idx)
            current_x, current_y, current_z = self.swarm._positions[self.uri_dict[drone_idx]]
            dist2goal = ((current_x - self.goal[drone_idx][0])**2 + (current_y - self.goal[drone_idx][1])**2 +(current_z - self.goal[drone_idx][2])**2 )**0.5
            print(f'distance to goal of drone {drone_idx} is : {dist2goal}')
            if dist2goal < 0.3:
                return 1
            else:
                return 0
        except:
            return 0

    def land(self, scf):
        cf = scf.cf
        commander = cf.high_level_commander
        commander.land(0.0, 4.0)
        time.sleep(4)
        commander.stop()


# --------------------------- Examples --------------------------------








# if __name__ == '__main__':
    # init mission
    

    # mission
    # swarm.parallel_safe(take_off)
    # t1 = swarm.trajectory_to_drone(execute_trajectory, uri1, waypoints= get_wp_circle(is_reversed=False))
    # open_threads.append(t1)
    # t2 = swarm.trajectory_to_drone(execute_trajectory, uri1, waypoints= get_wp(offset=(-0.35,-1),is_reversed=True))
    # open_threads.append(t2)

    # thread_running = True
    # while thread_running : # check if all threads are finished before sending new commands
    #     thread_running = False
    #     for thread in open_threads:
    #         if thread.is_alive():
    #             thread_running = True
        # print('main thread is running in background')
    #     time.sleep(0.1)
    # open_threads = []      
        
    # end mission
    # swarm.parallel_safe(land)
    # swarm.close_links()

