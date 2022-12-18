#! /usr/bin/env python3
import numpy as np
import params
import time
"""
how to use:
1. set is_reach_goal to 0.015 in cf mode
2. set limits of zone
3. set step_size [m]

"""
if params.mode == 'sim':
    from rotors_flight_manager import Flight_manager
    import rospy
    rospy.init_node('send_my_command', anonymous=True)
    rospy.sleep(3)
elif params.mode == 'cf':
    from CF_Flight_Manager import Flight_manager




def main():
    fc = Flight_manager(2)
    fc.take_off_swarm()

    time.sleep(5)
    goal = [-1, 0, 1]


    fc.go_to(drone_idx=0, goal=[-1, 0, 0.7])
    fc.go_to(drone_idx=1, goal=[-1, 0, 1.9])
    now = time.time()
    delta = 0
    while delta < 15:
        fc.go_to(drone_idx=0, goal=[-1, 0, 0.7])
        fc.go_to(drone_idx=1, goal=[-1, 0, 1.7])
        fc.sleep()
        delta = time.time() - now

                    
    fc.go_to(drone_idx=1, goal=[-1, 1, 1.7])              
    time.sleep(3)
    fc.land(drone_idx='all')

                
if __name__ == '__main__':
    main()

