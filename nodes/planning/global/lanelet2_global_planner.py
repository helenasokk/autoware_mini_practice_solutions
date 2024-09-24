#!/usr/bin/env python3

# All these imports from lanelet2 library should be sufficient
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

import rospy

from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Lane

class Lanelet2GlobalPlanner:
    def __init__(self):

        self.goal_point = None

        self.lanelet2_map_name = rospy.get_param('~lanelet2_map_name')
        self.coordinate_transformer = rospy.get_param('/localization/coordinate_transformer')
        self.use_custom_origin = rospy.get_param('/localization/use_custom_origin')
        self.utm_origin_lat = rospy.get_param('/localization/utm_origin_lat')
        self.utm_origin_lon = rospy.get_param('/localization/utm_origin_lon')

        if self.coordinate_transformer == "utm":
            projector = UtmProjector(Origin(self.utm_origin_lat, self.utm_origin_lon), self.use_custom_origin, True)
        else:
            raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + self.coordinate_transformer)
        
        self.lanelet2_map = load(self.lanelet2_map_name, projector)


        # Publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        self.route_pub = rospy.Publisher('global_path', Lane)#TODO

        # Subscribers
        #rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.routing, queue_size=1)

    
    def routing(self, msg):
        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, 
                                                      lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        if self.goal_point == None:
            return
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        # find routing graph
        try:
            route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)
        except:
            rospy.logwarn("No route has been found.")
            return None
        # find shortest path
        path = route.shortestPath()
        # this returns LaneletSequence to a point where lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(start_lanelet)
        print(path_no_lane_change)
    
    def goal_callback(self, msg):
        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        goal = PoseStamped()
        goal.header.stamp = msg.header.stamp
        goal.pose.position.x = msg.pose.position.x
        goal.pose.position.y = msg.pose.position.y
        goal.pose.position.z = msg.pose.position.z
        goal.pose.orientation.x = msg.pose.orientation.x
        goal.pose.orientation.y = msg.pose.orientation.y
        goal.pose.orientation.z = msg.pose.orientation.z
        goal.pose.orientation.w = msg.pose.orientation.w
        self.goal_pub.publish(goal)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()