#!/usr/bin/env python3

import rospy
import numpy as np

from autoware_msgs.msg import Lane, VehicleCmd
from geometry_msgs.msg import PoseStamped, Twist
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion
from scipy.interpolate import interp1d

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.path_linestring = None
        self.velocity_interpolator = None
        self.stop_requested = False
        # Reading in the parameter values
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.wheel_base = rospy.get_param("/vehicle/wheel_base")
        
        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=1)

        # Subscribers
        rospy.Subscriber('path', Lane, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):

        if not msg.waypoints or len(msg.waypoints) < 2:
            rospy.logwarn("Empty or invalid path. Stopping the car.")
            self.stop_requested = True
            return
        self.stop_requested = False
        # convert waypoints to shapely linestring
        self.path_linestring = LineString([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        # prepare path - creates spatial tree, making the spatial queries more efficient
        prepare(self.path_linestring)
        
        waypoints_xy = np.array([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
        
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0) # add 0 distance at the start
        
        self.velocity_interpolator = interp1d(distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)

    def stop_vehicle(self, msg):
        stop_msg = VehicleCmd()
        stop_msg.header.stamp = msg.header.stamp
        stop_msg.header.frame_id = "base_link"
        stop_msg.ctrl_cmd.steering_angle = 0.0
        stop_msg.ctrl_cmd.linear_velocity = 0.0
        self.vehicle_cmd_pub.publish(stop_msg)

    def current_pose_callback(self, msg):
        if self.stop_requested:
            self.stop_vehicle(msg)
            return
        if self.path_linestring == None or self.velocity_interpolator == None:
            return
        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        # Calculate the lookahead point and then the heading; the latter can be calculated from point coordinates using the arctan2 function.
        d_ego_from_path_start = self.path_linestring.project(current_pose)
        lookahead_point = self.path_linestring.interpolate(d_ego_from_path_start + self.lookahead_distance)

        lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)
        
        # Recalculate the lookahead distance - ld.
        self.lookahead_distance = current_pose.distance(lookahead_point)
        alpha = lookahead_heading - heading
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))# normalizing alpha
        
        
        steering_angle = np.arctan((2*self.wheel_base*np.sin(alpha))/self.lookahead_distance)

        
        velocity = self.velocity_interpolator(d_ego_from_path_start)
        
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = "base_link"
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
        vehicle_cmd.ctrl_cmd.linear_velocity = velocity
        self.vehicle_cmd_pub.publish(vehicle_cmd)
    	
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()
