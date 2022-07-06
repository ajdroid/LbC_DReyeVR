#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
All intersection related scenarios that are part of a route.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (TrafficLightManipulator,
                                                                      AccelerateToVelocity,
                                                                      TrafficLightStateSetter,
                                                                      StopVehicle)

from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, DrivenDistanceTest, MaxVelocityTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, WaitEndIntersection
from srunner.scenarios.basic_scenario import BasicScenario

from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      WaypointFollower,
                                                                      SyncArrival)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, DrivenDistanceTest, MaxVelocityTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (get_crossing_point,
                                           get_geometric_linear_intersection,
                                           generate_target_waypoint_list)


class SignalJunctionCrossingRoute(BasicScenario):

    """
    At routes, these scenarios are simplified, as they can be triggered making
    use of the background activity. To ensure interactions with this background
    activity, the traffic lights are modified, setting two of them to green
    """

    # ego vehicle parameters
    _ego_max_velocity_allowed = 20       # Maximum allowed velocity [m/s]
    _ego_expected_driven_distance = 50   # Expected driven distance [m]
    _ego_distance_to_traffic_light = 5  # Trigger distance to traffic light [m]
    _ego_distance_to_drive = 20          # Allowed distance to drive

    # other vehicle
    _other_actor_target_velocity = 10  # Target velocity of other vehicle
    _other_actor_max_brake = 1.0  # Maximum brake of other vehicle
    _other_actor_distance = 50  # Distance the other vehicle should drive

    _traffic_light = None

    # Depending on the route, decide which traffic lights can be modified

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        # Timeout of scenario in seconds
        self.timeout = timeout
        self.subtype = config.subtype
        self._reference_waypoint = CarlaDataProvider.get_map().get_waypoint(config.trigger_points[0].location)
        self._ego_traffic_light = CarlaDataProvider.get_next_traffic_light_by_location(config.trigger_points[0].location)

        _traffic_light = None

        super(SignalJunctionCrossingRoute, self).__init__("SignalJunctionCrossingRoute",
                                                          ego_vehicles,
                                                          config,
                                                          world,
                                                          debug_mode,
                                                          criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # adding the other_actor spawning behaviour here
        # from the non-route things
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        Scenario behavior:
        When close to an intersection, the traffic lights will turn green for
        the ego_vehicle and another lane, allowing the background activity
        to "run" their red light, creating scenarios 7, 8 and 9.

        If this does not happen within 120 seconds, a timeout stops the scenario
        """
        crossing_point_dynamic = get_crossing_point(self._reference_waypoint)
        print(crossing_point_dynamic.__str__())

        # start condition
        startcondition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            crossing_point_dynamic,
            self._ego_distance_to_traffic_light,
            name="Waiting for start position")

        other_movement = py_trees.composites.Parallel(
            "Move other actor to next intersection",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0],
                                                      1.0,
                                                      self._other_actor_target_velocity,
                                                      name="other actor start moving")
        actor_cross_lane = DriveDistance(self.other_actors[0],
                                         5,
                                         name="other actor drive distance for intersection crossing")
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0],
                                              self._other_actor_max_brake,
                                              name="other actor stop")
        other_movement.add_child(actor_start_cross_lane)
        other_movement.add_child(actor_cross_lane)
        other_movement.add_child(actor_stop_crossed_lane)

        # Generate plan for WaypointFollower
        turn = 0 # drive straight ahead
        plan = []
        # generating waypoints until intersection (target_waypoint)
        plan, target_waypoint = generate_target_waypoint_list(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), turn)

        # Generating waypoint list till next intersection
        wp_choice = target_waypoint.next(5.0)
        while len(wp_choice) == 1:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(5.0)

        continue_driving = py_trees.composites.Parallel(
            "ContinueDriving",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        continue_driving_waypoints = WaypointFollower(
            self.other_actors[0], self._other_actor_target_velocity, plan=plan, avoid_collision=False)

        continue_driving_distance = DriveDistance(
            self.other_actors[0],
            self._other_actor_distance,
            name="Distance")

        continue_driving_timeout = TimeOut(10)

        continue_driving.add_child(continue_driving_waypoints)
        continue_driving.add_child(continue_driving_distance)
        continue_driving.add_child(continue_driving_timeout)

        # Changes traffic lights
        traffic_hack = TrafficLightManipulator(self.ego_vehicles[0], self.subtype)

        # finally wait that ego vehicle drove a specific distance
        wait = DriveDistance(
            self.ego_vehicles[0],
            self._ego_distance_to_drive,
            name="DriveDistance")

        ego_light_green = TrafficLightStateSetter(
            actor=self._ego_traffic_light,
            state=carla.TrafficLightState.Green,
            name="Change ego light to green")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("SignalJunctionCrossingRoute")
        # sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(startcondition)
        sequence.add_child(ego_light_green)
        # sequence.add_child(traffic_hack)
        # sequence.add_child(other_movement)
        sequence.add_child(continue_driving)
        sequence.add_child(wait)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(wait)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicles[0],
            self._ego_max_velocity_allowed,
            optional=True)
        collision_criterion = CollisionTest(self.ego_vehicles[0])
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicles[0],
            self._ego_expected_driven_distance)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(driven_distance_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self._traffic_light = None
        self.remove_all_actors()


class NoSignalJunctionCrossingRoute(BasicScenario):

    """
    At routes, these scenarios are simplified, as they can be triggered making
    use of the background activity. For unsignalized intersections, just wait
    until the ego_vehicle has left the intersection.
    """

    # ego vehicle parameters
    _ego_max_velocity_allowed = 20       # Maximum allowed velocity [m/s]
    _ego_expected_driven_distance = 50   # Expected driven distance [m]
    _ego_distance_to_drive = 20          # Allowed distance to drive

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(NoSignalJunctionCrossingRoute, self).__init__("NoSignalJunctionCrossingRoute",
                                                            ego_vehicles,
                                                            config,
                                                            world,
                                                            debug_mode,
                                                            criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

    def _create_behavior(self):
        """
        Scenario behavior:
        When close to an intersection, the traffic lights will turn green for
        both the ego_vehicle and another lane, allowing the background activity
        to "run" their red light.

        If this does not happen within 120 seconds, a timeout stops the scenario
        """
        # finally wait that ego vehicle drove a specific distance
        wait = WaitEndIntersection(
            self.ego_vehicles[0],
            name="WaitEndIntersection")
        end_condition = DriveDistance(
            self.ego_vehicles[0],
            self._ego_distance_to_drive,
            name="DriveDistance")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("NoSignalJunctionCrossingRoute")
        sequence.add_child(wait)
        sequence.add_child(end_condition)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicles[0],
            self._ego_max_velocity_allowed,
            optional=True)
        collision_criterion = CollisionTest(self.ego_vehicles[0])
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicles[0],
            self._ego_expected_driven_distance)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(driven_distance_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
