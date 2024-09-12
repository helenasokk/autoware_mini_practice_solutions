#!/usr/bin/env python3

import math
import rospy

from tf.transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from pyproj import CRS, Transformer, Proj

from novatel_oem7_msgs.msg import INSPVA
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion, TransformStamped

class Localizer:
    def __init__(self):

        # Parameters
        self.undulation = rospy.get_param('undulation')
        utm_origin_lat = rospy.get_param('utm_origin_lat')
        utm_origin_lon = rospy.get_param('utm_origin_lon')

        # Internal variables
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_utm = CRS.from_epsg(25835)
        self.utm_projection = Proj(self.crs_utm)

        # Subscribers
        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.transform_coordinates)
        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.transform_velocity)

        # Publishers
        self.current_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=10)
        self.current_velocity_pub = rospy.Publisher('current_velocity', TwistStamped, queue_size=10)
        self.br = TransformBroadcaster()
        
        # create coordinate transformer
        self.transformer = Transformer.from_crs(self.crs_wgs84, self.crs_utm)
        self.origin_x, self.origin_y = self.transformer.transform(utm_origin_lat, utm_origin_lon)
        

    # convert azimuth to yaw angle
    def convert_azimuth_to_yaw(self, azimuth):
        yaw = -azimuth + math.pi/2
        # Clamp within 0 to 2 pi
        if yaw > 2 * math.pi:
           yaw = yaw - 2 * math.pi
        elif yaw < 0:
           yaw += 2 * math.pi

        return yaw
        
    def transform_coordinates(self, msg):
        latitude, longitude = self.transformer.transform(msg.latitude, msg.longitude)
        print(latitude - self.origin_x, longitude - self.origin_y)
        
                # calculate azimuth correction
        azimuth_correction = self.utm_projection.get_factors(msg.longitude,msg.latitude).meridian_convergence
        
        azimuth = msg.azimuth - azimuth_correction
        
        yaw = self.convert_azimuth_to_yaw(azimuth)
        
        # Convert yaw to quaternion
        x, y, z, w = quaternion_from_euler(0, 0, yaw)
        orientation = Quaternion(x, y, z, w)
        
        # publish current pose
        current_pose_msg = PoseStamped()
        current_pose_msg.header.stamp = msg.header.stamp
        current_pose_msg.header.frame_id = 'map'
        current_pose_msg.pose.position.x = latitude - self.origin_x
        current_pose_msg.pose.position.y = longitude - self.origin_y
        current_pose_msg.pose.position.z = msg.height - self.undulation
        current_pose_msg.pose.orientation = orientation
        self.current_pose_pub.publish(current_pose_msg)
        
        self.publish_transform(current_pose_msg)
        
    def publish_transform(self, pose_msg):
    	t = TransformStamped()
    	
    	t.header.stamp = pose_msg.header.stamp
    	t.header.frame_id = "map"
    	t.child_frame_id = "base_link"
    	
    	t.transform.translation.x = pose_msg.pose.position.x
    	t.transform.translation.y = pose_msg.pose.position.y
    	t.transform.translation.z = pose_msg.pose.position.z
    	
    	t.transform.rotation = pose_msg.pose.orientation
    	
    	self.br.sendTransform(t)
        
    def transform_velocity(self, msg):
        # Calculate velocity norm
        calc_vel = math.sqrt(msg.north_velocity ** 2 + msg.east_velocity ** 2)
        # Create TwistStamped message
        current_vel_msg = TwistStamped()
        current_vel_msg.header.stamp = msg.header.stamp
        current_vel_msg.header.frame_id = "base_link"
        current_vel_msg.twist.linear.x = calc_vel
        
        #Publish
        self.current_velocity_pub.publish(current_vel_msg)
        

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('localizer')
    node = Localizer()
    node.run()
