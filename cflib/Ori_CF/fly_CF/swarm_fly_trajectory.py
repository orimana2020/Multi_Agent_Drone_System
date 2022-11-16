import time
import cflib.crtp
import numpy as np
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.Ori_CF.fly_CF.Trajectory import Generate_Trajectory, upload_trajectory


def activate_high_level_commander(scf):
    scf.cf.param.set_value('commander.enHighLevel', '1')


def activate_mellinger_controller(scf, use_mellinger):
    controller = 1
    if use_mellinger:
        controller = 2
    scf.cf.param.set_value('stabilizer.controller', controller)


def run_shared_sequence(scf):
    box_size = 1
    flight_time = 2
    commander = scf.cf.high_level_commander
    commander.takeoff(1.0, 2.0)
    time.sleep(3)
    commander.go_to(box_size, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)
    commander.go_to(0, box_size, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)
    commander.go_to(-box_size, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)
    commander.go_to(0, -box_size, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)
    commander.land(0.0, 2.0)
    time.sleep(2)
    commander.stop()





def get_wp(offset, is_reversed=False):
    x_off,y_off = offset
    wp_orig = []
    for x in range(10):
        wp_orig.append([x_off+0.1*x, y_off+x*0.1, 1]) 
    wp_orig = np.array(wp_orig)
    if is_reversed:
        return wp_orig[::-1]
    return wp_orig

def take_off(scf):
    cf = scf.cf
    commander = cf.high_level_commander
    commander.takeoff(1.0, 2.0)
    time.sleep(2.0) 

def execute_trajectory(scf, waypoints): 
    cf = scf.cf
    commander = cf.high_level_commander  
    try:
        trajectory_id = 1
        traj = Generate_Trajectory(waypoints, velocity=1, plotting=0)
        traj_coef = traj.poly_coef
        duration = upload_trajectory(cf, trajectory_id ,traj_coef)
        commander.start_trajectory(trajectory_id, 1.0, False)
        time.sleep(duration)
    except:
        print('failed to execute trajectory')
    

def land(scf):
    cf = scf.cf
    commander = cf.high_level_commander
    commander.land(0.0, 2.0)
    time.sleep(2)
    commander.stop()

# def run_sequence2(scf):
#     cf = scf.cf
#     commander = cf.high_level_commander
#     # go forward
#     execute_trajectory(cf, commander, wp = get_wp(offset=(-0.2,0.5)))
#     # go backward
#     execute_trajectory(cf, commander, wp = get_wp(offset=(-0.5,0.5), is_reversed=True))



uri1 = 'radio://0/80/2M/E7E7E7E7E1'
uri2 = 'radio://0/80/2M/E7E7E7E7E2'
uri3 = 'radio://0/80/2M/E7E7E7E7E3'
uri4 = 'radio://0/80/2M/E7E7E7E7E4'
uri_list = [uri4, uri1]
uris = set(uri_list)
print(uris)
open_threads = []

if __name__ == '__main__':
    # init mission
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    swarm = Swarm(uris, factory=factory)
    swarm.open_links()
    swarm.parallel_safe(activate_high_level_commander)
    swarm.reset_estimators()

    # mission
    swarm.parallel_safe(take_off)
    t1 = swarm.trajectory_to_drone(execute_trajectory, uri4, waypoints= get_wp(offset=(-0.35,-0.5),is_reversed=True))
    open_threads.append(t1)
    t2 = swarm.trajectory_to_drone(execute_trajectory, uri1, waypoints= get_wp(offset=(-0.35,-1),is_reversed=True))
    open_threads.append(t2)

    thread_running = True
    while thread_running : #must check if previous thread is finished before sending new commands
        thread_running = False
        for thread in open_threads:
            if thread.is_alive():
                thread_running = True
        print('main thread is running in background')
        time.sleep(0.1)
    open_threads = []      
        
    # end mission
    swarm.parallel_safe(land)
    swarm.close_links()

