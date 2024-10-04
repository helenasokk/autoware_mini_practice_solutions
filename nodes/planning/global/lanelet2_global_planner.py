#!/usr/bin/env python3

import math
# All these imports from lanelet2 library should be sufficient
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

from shapely.geometry import Point, LineString

import rospy

from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Lane, Waypoint

class Lanelet2GlobalPlanner:
    def __init__(self):

        self.goal_point = None
        self.goal_reached = False

        self.lanelet2_map_name = rospy.get_param('~lanelet2_map_name')
        self.speed_limit = rospy.get_param('~speed_limit', 40.0)
        self.output_frame = rospy.get_param('/planning/lanelet2_global_planner/output_frame', 'map')
        self.distance_to_goal_limit = rospy.get_param('/planning/lanelet2_global_planner/distance_to_goal_limit', 5.0)
        self.coordinate_transformer = rospy.get_param('/localization/coordinate_transformer')
        self.use_custom_origin = rospy.get_param('/localization/use_custom_origin')
        self.utm_origin_lat = rospy.get_param('/localization/utm_origin_lat')
        self.utm_origin_lon = rospy.get_param('/localization/utm_origin_lon')

        if self.coordinate_transformer == "utm":
            projector = UtmProjector(Origin(self.utm_origin_lat, self.utm_origin_lon), self.use_custom_origin, True)
        else:
            raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + self.coordinate_transformer)
        
        self.lanelet2_map = load(self.lanelet2_map_name, projector)

        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, 
                                                      lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)


        # Publishers
        self.waypoints_pub = rospy.Publisher('global_path', Lane, queue_size=1, latch=True)
        self.route_pub = rospy.Publisher('global_path', Lane, queue_size=1)

        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.routing, queue_size=1)

    def distance_to_goal(self, current_pose):
        if self.goal_point is None:
            return float('inf')
        
        dx = current_pose.x - self.goal_point.x
        dy = current_pose.y - self.goal_point.y
        return math.sqrt(dx * dx + dy * dy)
    
    def routing(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        if self.goal_point == None:
            #rospy.logwarn("Goal point not set!")
            return
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        # find routing graph
        try:
            route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)
            if not route:
                rospy.logwarn("No route found.")
                return
            # find shortest path
            path = route.shortestPath()
            if not path:
                rospy.logwarn("No path found.")
                return

            distance = self.distance_to_goal(self.current_location)
            if distance < self.distance_to_goal_limit:
                if not self.goal_reached:
                    rospy.loginfo("Goal reached, clearing path")
                    self.clear_path(msg)
                    self.goal_reached = True
                return
            self.goal_reached = False
            # this returns LaneletSequence to a point wexcept:
            path_no_lane_change = path.getRemainingLane(start_lanelet)
            if not path_no_lane_change:
                rospy.logwarn("No path found without lane change.")
                return
            print(path_no_lane_change)
            projected_goal = self.project_goal_on_path(path_no_lane_change[-1].centerline, self.goal_point)

            global_path = self.convert_to_lane_msg(msg, path_no_lane_change)
            self.route_pub.publish(global_path)

            waypoints = self.lanelet_seq_2_waypoints(path_no_lane_change, projected_goal)
            self.publish_global_path(waypoints)
        except:
            rospy.logwarn("No route has been found.")
            return
        
    def project_goal_on_path(self, lanelet_center, goal_point):
        # Tried to implement so that the goal would be the last waypoint of the path
        # project the goal point onto the nearest location on the lanelet centerline
        centerline_coords = [(p.x, p.y) for p in lanelet_center]
        line = LineString(centerline_coords)

        goal = Point(goal_point.x, goal_point.y)

        projected_dist = line.project(goal)
        projected_point = line.interpolate(projected_dist)

        return BasicPoint2d(projected_point.x, projected_point.y)
        
    def clear_path(self, msg):
        lane = Lane()
        lane.header.frame_id = self.output_frame
        lane.header.stamp = msg.header.stamp
        lane.waypoints = []

        self.waypoints_pub.publish(lane)

    def convert_to_lane_msg(self, msg, lanelet_seq):
        lane_msg = Lane()
        lane_msg.header.stamp = msg.header.stamp
        lane_msg.header.frame_id = msg.header.frame_id
        for lanelet in lanelet_seq:
            for point in lanelet.centerline:
                waypoint = Waypoint()
                waypoint.pose.pose.position.x = point.x
                waypoint.pose.pose.position.y = point.y
                waypoint.pose.pose.position.z = point.z
                lane_msg.waypoints.append(waypoint)
        return lane_msg
    
    def lanelet_seq_2_waypoints(self, path_no_lane_change, projected_goal):
        waypoints = []

        for lanelet in path_no_lane_change:
            if 'speed_ref' in lanelet.attributes:
                speed = float(lanelet.attributes['speed_ref'])*1000/3600
            else:
                speed = self.speed_limit*1000/3600

            speed = min(speed, self.speed_limit*1000/3600)
            centerline = lanelet.centerline

            for i, point in enumerate(lanelet.centerline):
                if len(waypoints) > 0 and i == 0:
                    continue

                waypoint = Waypoint()
                waypoint.pose.pose.position.x = point.x
                waypoint.pose.pose.position.y = point.y
                waypoint.pose.pose.position.z = point.z
                waypoint.twist.twist.linear.x = speed

                waypoints.append(waypoint)

            if lanelet == path_no_lane_change[-1]:
                projected_goal = self.project_goal_on_path(centerline, self.goal_point)
        
        if projected_goal:
            # The speed stays the same, I'm changing only the x and y position of the last waypoint
            waypoints[-1].pose.pose.position.x = projected_goal.x
            waypoints[-1].pose.pose.position.y = projected_goal.y

        return waypoints
    
    def publish_global_path(self, waypoints):
        lane = Lane()
        lane.header.frame_id = self.output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        self.waypoints_pub.publish(lane)


    
    def goal_callback(self, msg):
        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()