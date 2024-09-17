#!/usr/bin/env python3

import rospy
import numpy as np

from autoware_msgs.msg import Lane, VehicleCmd
from geometry_msgs.msg import PoseStamped
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.path_linestring = None
        # Reading in the parameter values
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.wheel_base = rospy.get_param("/vehicle/wheel_base")
        
        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd)

        # Subscribers
        rospy.Subscriber('path', Lane, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        # convert waypoints to shapely linestring
        path_linestring = LineString([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        # prepare path - creates spatial tree, making the spatial queries more efficient
        prepare(path_linestring)

    def current_pose_callback(self, msg):
        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # Calculate the lookahead point and then the heading; the latter can be calculated from point coordinates using the arctan2 function.
        lookahead_point = lookahead_distance.interpolate(heading)
        lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)
        # Recalculate the lookahead distance - ld.
        lookahead_distance = shapely.distance(lookahead_point, current_pose)
        if self.path_linestring == None:
            return
        d_ego_from_path_start = self.path_linestring.project(current_pose)
        print(d_ego_from_path_start)
        steering_angle = np.arctan2((2*wheel_base*np.sin(heading-lookahead_heading))/lookahead_distance)
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = "base_link"
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
        vehicle_cmd.ctrl_cmd.linear_velocity = 10.0
        self.vehicle_cmd_pub.publish(vehicle_cmd)
    	
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()
