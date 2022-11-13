# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2019 Bitcraze AB
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# ---------------- bug to fix -------------------
# yaw problem- example- going from -160 to 150 yaw deg - is going througth 0 and not in shortest path throuth 180 deg 
# because discountiniuty 180/-180 deg


"""
Example of how to generate trajectories for the High Level commander using
Bezier curves. The output from this script is intended to be pasted into the
autonomous_sequence_high_level.py example.

This code uses Bezier curves of degree 7, that is with 8 control points.
See https://en.wikipedia.org/wiki/B%C3%A9zier_curve

All coordinates are (x, y, z, yaw)
"""
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from scipy import interpolate

class Node:
    """
    A node represents the connection point between two Bezier curves
    (called Segments).
    It holds 4 control points for each curve and the positions of the control
    points are set to join the curves with continuity in c0, c1, c2, c3.
    See https://www.cl.cam.ac.uk/teaching/2000/AGraphHCI/SMEG/node3.html

    The control points are named
    p4, p5, p6 and p7 for the tail of the first curve
    q0, q1, q2, q3 for the head of the second curve
    """

    def __init__(self, q0, q1=None, q2=None, q3=None):
        """
        Create a Node. Pass in control points to define the shape of the
        two segments that share the Node. The control points are for the
        second segment, that is the four first control points of the Bezier
        curve after the node. The control points for the Bezier curve before
        the node are calculated from the existing control points.
        The control points are for scale = 1, that is if the Bezier curve
        after the node has scale = 1 it will have exactly these handles. If the
        curve after the node has a different scale the handles will be moved
        accordingly when the Segment is created.

        q0 is required, the other points are optional.
        if q1 is missing it will be set to generate no velocity in q0.
        If q2 is missing it will be set to generate no acceleration in q0.
        If q3 is missing it will be set to generate no jerk in q0.

        If only q0 is set, the node will represent a point where the Crazyflie
        has no velocity. Good for starting and stopping.

        To get a fluid motion between segments, q1 must be set.

        :param q0: The position of the node
        :param q1: The position of the first control point
        :param q2: The position of the second control point
        :param q3: The position of the third control point
        """
        self._control_points = np.zeros([2, 4, 4])

        q0 = np.array(q0)

        if q1 is None:
            q1 = q0
        else:
            q1 = np.array(q1)
            # print('q1 generated:', q1)

        d = q1 - q0

        if q2 is None:
            q2 = q0 + 2 * d
            # print('q2 generated:', q2)
        else:
            q2 = np.array(q2)

        e = 3 * q2 - 2 * q0 - 6 * d

        if q3 is None:
            q3 = e + 3 * d
            # print('q3 generated:', q3)
        else:
            q3 = np.array(q3)

        p7 = q0
        p6 = q1 - 2 * d
        p5 = q2 - 4 * d
        p4 = 2 * e - q3

        self._control_points[0][0] = q0
        self._control_points[0][1] = q1
        self._control_points[0][2] = q2
        self._control_points[0][3] = q3

        self._control_points[1][3] = p7
        self._control_points[1][2] = p6
        self._control_points[1][1] = p5
        self._control_points[1][0] = p4

    def get_head_points(self):
        return self._control_points[0]

    def get_tail_points(self):
        return self._control_points[1]

    def draw_unscaled_controlpoints(self, visualizer):
        color = (0.8, 0.8, 0.8)
        for p in self._control_points[0]:
            visualizer.marker(p[0:3], color=color)
        for p in self._control_points[1]:
            visualizer.marker(p[0:3], color=color)

    def print(self):
        print('Node ---')
        print('Tail:')
        for c in self._control_points[1]:
            print(c)
        print('Head:')
        for c in self._control_points[0]:
            print(c)


class Segment:
    """
    A Segment represents a Bezier curve of degree 7. It uses two Nodes to
    define the shape. The scaling of the segment will move the handles compared
    to the Node to maintain continuous position, velocity, acceleration and
    jerk through the Node.
    A Segment can generate a polynomial that is compatible with the High Level
    Commander, either in python to be sent to the Crazyflie, or as C code to be
    used in firmware.
    A Segment can also be rendered in Vispy.
    """

    def __init__(self, head_node, tail_node, scale):
        self._scale = scale

        unscaled_points = np.concatenate(
            [head_node.get_head_points(), tail_node.get_tail_points()])

        self._points = self._scale_control_points(unscaled_points, self._scale)

        polys = self._convert_to_polys()
        self._polys = self._stretch_polys(polys, self._scale)

        self._vel = self._deriv(self._polys)
        self._acc = self._deriv(self._vel)
        self._jerk = self._deriv(self._acc)

    def _convert_to_polys(self):
        n = len(self._points) - 1
        result = np.zeros([4, 8])

        for d in range(4):
            for j in range(n + 1):
                s = 0.0
                for i in range(j + 1):
                    s += ((-1) ** (i + j)) * self._points[i][d] / (
                        math.factorial(i) * math.factorial(j - i))

                c = s * math.factorial(n) / math.factorial(n - j)
                result[d][j] = c

        return result

    def draw_trajectory(self, visualizer):
        self._draw(self._polys, 'black', visualizer)

    def draw_vel(self, visualizer):
        self._draw(self._vel, 'green', visualizer)

    def draw_acc(self, visualizer):
        self._draw(self._acc, 'red', visualizer)

    def draw_jerk(self, visualizer):
        self._draw(self._jerk, 'blue', visualizer)

    def draw_control_points(self, visualizer):
        for p in self._points:
            visualizer.marker(p[0:3])

    def _draw(self, polys, color, visualizer):
        step = self._scale / 32
        prev = None
        for t in np.arange(0.0, self._scale + step, step):
            p = self._eval_xyz(polys, t)

            if prev is not None:
                visualizer.line(p, prev, color=color)

            prev = p

    def velocity(self, t):
        return self._eval_xyz(self._vel, t)

    def acceleration(self, t):
        return self._eval_xyz(self._acc, t)

    def jerk(self, t):
        return self._eval_xyz(self._jerk, t)

    def _deriv(self, p):
        result = []
        for i in range(3):
            result.append([
                1 * p[i][1],
                2 * p[i][2],
                3 * p[i][3],
                4 * p[i][4],
                5 * p[i][5],
                6 * p[i][6],
                7 * p[i][7],
                0
            ])

        return result

    def _eval(self, p, t):
        result = 0
        for part in range(8):
            result += p[part] * (t ** part)
        return result

    def _eval_xyz(self, p, t):
        return np.array(
            [self._eval(p[0], t), self._eval(p[1], t), self._eval(p[2], t)])

    def print_poly_python(self):
        s = '  [' + str(self._scale) + ', '

        for axis in range(4):
            for d in range(8):
                s += str(self._polys[axis][d]) + ', '

        s += '],  # noqa'
        print(s)
    
    def get_coef(self):
        coef = []
        coef.append(self._scale)
        for axis in range(4):
            for d in range(8):
                coef.append(self._polys[axis][d]) 
        return coef


    def print_poly_c(self):
        s = ''

        for axis in range(4):
            for d in range(8):
                s += str(self._polys[axis][d]) + ', '

        s += str(self._scale)
        print(s)

    def print_points(self):
        print(self._points)

    def print_peak_vals(self):
        peak_v = 0.0
        peak_a = 0.0
        peak_j = 0.0

        step = 0.05
        for t in np.arange(0.0, self._scale + step, step):
            peak_v = max(peak_v, np.linalg.norm(self._eval_xyz(self._vel, t)))
            peak_a = max(peak_a, np.linalg.norm(self._eval_xyz(self._acc, t)))
            peak_j = max(peak_j, np.linalg.norm(self._eval_xyz(self._jerk, t)))

        print('Peak v:', peak_v, 'a:', peak_a, 'j:', peak_j)

    def _stretch_polys(self, polys, time):
        result = np.zeros([4, 8])

        recip = 1.0 / time

        for p in range(4):
            scale = 1.0
            for t in range(8):
                result[p][t] = polys[p][t] * scale
                scale *= recip

        return result

    def _scale_control_points(self, unscaled_points, scale):
        s = scale
        l_s = 1 - s
        p = unscaled_points

        result = [None] * 8

        result[0] = p[0]
        result[1] = l_s * p[0] + s * p[1]
        result[2] = l_s ** 2 * p[0] + 2 * l_s * s * p[1] + s ** 2 * p[2]
        result[3] = l_s ** 3 * p[0] + 3 * l_s ** 2 * s * p[
            1] + 3 * l_s * s ** 2 * p[2] + s ** 3 * p[3]
        result[4] = l_s ** 3 * p[7] + 3 * l_s ** 2 * s * p[
            6] + 3 * l_s * s ** 2 * p[5] + s ** 3 * p[4]
        result[5] = l_s ** 2 * p[7] + 2 * l_s * s * p[6] + s ** 2 * p[5]
        result[6] = l_s * p[7] + s * p[6]
        result[7] = p[7]

        return result



# segment_time = 2
# z = 1
# yaw = 0

# segments = []

# Nodes with one control point has not velocity, this is similar to calling
# goto in the High-level commander

# n0 = Node((0, 0, z, yaw))
# n1 = Node((1, 0, z, yaw))
# n2 = Node((1, 1, z, yaw))
# n3 = Node((0, 1, z, yaw))

# segments.append(Segment(n0, n1, segment_time))
# segments.append(Segment(n1, n2, segment_time))
# segments.append(Segment(n2, n3, segment_time))
# segments.append(Segment(n3, n0, segment_time))


# By setting the q1 control point we get velocity through the nodes
# Increase d to 0.7 to get some more action
# d = 0.1

# n5 = Node((1, 0, z, yaw),  )
# n6 = Node((1, 1, z, yaw), q1=(1 - d, 1 + d, z, yaw))
# n7 = Node((0, 1, z, yaw), q1=(0 - d, 1 - d, z, yaw))

# segments.append(Segment(n0, n5, segment_time))
# segments.append(Segment(n5, n6, segment_time))
# segments.append(Segment(n6, n7, segment_time))
# segments.append(Segment(n7, n0, segment_time))


# When setting q2 we can also control acceleration and get more action.
# Yaw also adds to the fun.

# d2 = 0.2
# dyaw = 2
# f = -0.3

# n8 = Node(
#     (1, 0, z, yaw),
#     q1=(1 + d2, 0 + d2, z, yaw),
#     q2=(1 + 2 * d2, 0 + 2 * d2 + 0*f * d2, 1, yaw))
# n9 = Node(
#     (1, 1, z, yaw + dyaw),
#     q1=(1 - d2, 1 + d2, z, yaw + dyaw),
#     q2=(1 - 2 * d2 + f * d2, 1 + 2 * d2 + f * d2, 1, yaw + dyaw))
# n10 = Node(
#     (0, 1, z, yaw - dyaw),
#     q1=(0 - d2, 1 - d2, z, yaw - dyaw),
#     q2=(0 - 2 * d2,  1 - 2 * d2,  1, yaw - dyaw))

# segments.append(Segment(n0, n8, segment_time))
# segments.append(Segment(n8, n9, segment_time))
# segments.append(Segment(n9, n10, segment_time))
# segments.append(Segment(n10, n0, segment_time))

# -------------------- my addition

def cntl_pnt(current, next):
    dic_vec = next - current
    size = np.linalg.norm(dic_vec, ord=2)
    normilized_dir_vec = dic_vec / size
    size_cp = size / 8
    return current + normilized_dir_vec * size_cp



def get_smooth_path(path):
    tck, _ = interpolate.splprep([path[:,0], path[:,1], path[:,2]], s=10)  
    u_fine = np.linspace(0,1,int(min(len(path), 30))) # determine number of points in smooth path 
    smooth_path = interpolate.splev(u_fine, tck)
    return np.transpose(np.array(smooth_path))

def get_segments_time(wp,velocity=1):
    segment_time = []
    for i in range(len(wp)-1):
        dist = np.linalg.norm(wp[i+1]-wp[i], ord=2)
        t = dist / velocity
        segment_time.append(t)
    return segment_time

def get_yaw(wp):
    yaws = []
    for i in range(len(wp)-1):
        dx = wp[i+1][0] - wp[i][0]
        dy = wp[i+1][1]- wp[i][1]
        yaw = np.arctan2(dy,dx)
        yaws.append(yaw)
    return yaws


def generate_nodes(wp, yaw):
    # init
    nodes = []
    wp1 = wp[0]
    node1 = Node((wp1[0], wp1[1], wp1[2], yaw[0]))
    nodes.append(node1)

    for i in range(1, len(wp)-1):
        wp1 = wp[i]
        next = wp[i+1]
        cp = cntl_pnt(wp1, next)
        node = Node((wp1[0], wp1[1], wp1[2], yaw[i]), q1=(cp[0], cp[1], cp[2], yaw[i]))
        nodes.append(node)
    # end
    wp1 = wp[-1]
    node = Node((wp1[0], wp1[1], wp1[2], yaw[-1]))
    nodes.append(node)
    return nodes

def generate_segments(nodes, segments_time):
    segments = []
    for i in range(len(nodes)-1):
        segments.append(Segment(nodes[i], nodes[i+1], segments_time[i]))
    return segments


def get_polynom_coeff(segments): # paste poly_coef in high level commander
    poly_coef = []
    for s in segments:
        poly_coef.append(s.get_coef())
    return poly_coef


def plot_trajectory(poly_coef, wp ,wp_orig, res):
    poly_coef = np.array(poly_coef)
    x_tot = np.array([])
    y_tot = np.array([])
    z_tot = np.array([])
    yaw_tot = np.array([])
    for segment in range(len(poly_coef)):
        row = poly_coef[segment]
        durition = row[0]
        x_c = row[1:9]
        y_c = row[9:17]
        z_c = row[17:25]
        yaw_c = row[25:]
        t_vec = np.linspace(0,durition, res)
        x = 0
        y = 0
        z = 0
        yaw = 0
        for i in range(len(x_c)):
            x += x_c[i]*t_vec**[i]
            y += y_c[i]*t_vec**[i]
            z += z_c[i]*t_vec**[i]
            yaw += yaw_c[i]*t_vec**[i]
        x_tot = np.append(x_tot, x, axis = None)
        y_tot = np.append(y_tot, y, axis = None)
        z_tot = np.append(z_tot, z, axis = None)
        yaw_tot = np.append(yaw_tot, yaw,axis= None)
    
    # yaw decoder
    r = 0.1
    # yaw_rad = np.deg2rad(yaw_tot)
    dy = r * np.sin(yaw_tot)
    dx = r * np.cos(yaw_tot)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(yaw_tot)):
        ax.plot([x_tot[i], x_tot[i]+dx[i] ], [y_tot[i], y_tot[i]+ dy[i] ], [z_tot[i], z_tot[i]], c='c')
    ax.scatter3D(x_tot, y_tot, z_tot)
    ax.scatter3D(wp_orig[:,0],wp_orig[:,1],wp_orig[:,2],s=40,c='green')
    ax.scatter3D(wp[:,0],wp[:,1],wp[:,2],s=40,c='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(0,10)
    ax.set_ylim(-5,5)
    ax.set_zlim(0,5)
    plt.show()



# ---------------------------------------------- examples

def get_original_wp_line():
    wp_orig = []
    for x in range(10):
        if x%2 == 0:
            y = 0
        elif x%3 == 0:
            y=2
        else:
            y = 1
        wp_orig.append([x,y,1]) 
    wp_orig = np.array(wp_orig)
    return wp_orig



def get_original_wp_spiral():
    rad = 0.7
    t = np.linspace(0, 3*2*np.pi, 20)
    wp_orig = np.array([[rad * np.sin(t[0]), rad * np.cos(t[0]), t[0]/6]])
    for i in range(1, len(t)):
        wp_orig = np.append(wp_orig, np.array([[rad * np.sin(t[i]), rad * np.cos(t[i]), t[i]/6]]) ,axis=0) 
    return wp_orig


if __name__ == '__main__':
    mode = 'Astar interpolaton' #'Astar interpolaton'

    wp_orig = get_original_wp_line()
    velocity = 1 #[m/s]
    if mode == 'Astar interpolaton':
        wp = get_smooth_path(wp_orig)
    else:
        wp = wp_orig
    yaw = get_yaw(wp)
    nodes = generate_nodes(wp, yaw)
    segments_time = get_segments_time(wp,velocity)
    segments = generate_segments(nodes, segments_time)
    poly_coef = get_polynom_coeff(segments)
    plot_trajectory(poly_coef,wp,wp_orig, res = 7)











